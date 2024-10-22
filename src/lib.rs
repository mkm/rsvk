#![feature(try_blocks)]
#![feature(never_type)]
#![feature(exhaustive_patterns)]

use std::borrow::Borrow;
use std::collections::VecDeque;
use std::error::Error;
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{
        swapchain::SwapchainImage,
        view::{ImageView, ImageViewCreateInfo},
        ImageAccess, ImageAspects, ImageSubresourceRange,
    },
    pipeline::{
        graphics::viewport::{Scissor, Viewport},
        Pipeline, PipelineBindPoint,
    },
    render_pass::FramebufferCreateInfo,
    swapchain::{self, PresentInfo, PresentMode, Surface, Swapchain, SwapchainCreateInfo},
    sync::{FenceSignalFuture, GpuFuture},
};
use xcb::{x, Xid};

#[macro_use]
mod prelude;
#[macro_use]
mod react;
mod gpu;
mod janitor;
mod maths;
mod poll;
mod render;
mod shaders;
mod stopwatch;

use gpu::Gpu;
use janitor::Janitor;
use maths::{Matrix3f, Matrix4f, Vector2f};
use prelude::*;
use react::{CachedResultNode, ConstNode, Source};
use render::{Framebuffer, GraphicsPipeline, RenderPass};
use shaders::passes::SimplePass;
use shaders::pipelines;
use stopwatch::Stopwatch;

#[derive(Clone)]
struct Frame {
    chain: Arc<Swapchain<()>>,
    images: Vec<Arc<SwapchainImage<()>>>,
}

struct Window {
    handle: ConstNode<x::Window>,
    surface: CachedResultNode<Arc<Surface<()>>, E>,
    surface_format: CachedResultNode<Format, E>,
    frame: Option<Frame>,
    extent: ConstNode<Option<[u32; 2]>>,
}

struct App {
    running: bool,
    conn: ConstNode<Arc<xcb::Connection>>,
    window: Window,
    gpu: CachedResultNode<Gpu, E>,
    janitor: Janitor,
    simple_pass: CachedResultNode<RenderPass<SimplePass>, E>,
    player_pipeline: CachedResultNode<GraphicsPipeline<pipelines::player::Pipeline>, E>, // Box<CreatePlayerPipeline>,
    present_future: Option<Arc<FenceSignalFuture<Box<dyn GpuFuture + Send>>>>,
    needs_render: bool,
    num_samples: i32,
    position: Vector2f,
    movements: VecDeque<(Instant, Vector2f)>,
}

fn init_surface(
    gpu: Gpu,
    conn: &xcb::Connection,
    window: &x::Window,
) -> R<(Arc<Surface<()>>, Format)> {
    let surface: Arc<Surface<()>> = unsafe {
        Surface::from_xcb(
            gpu.device.instance().clone(),
            conn.get_raw_conn(),
            window.resource_id(),
            (),
        )?
    };

    let (surface_format, _surface_colour_space) = gpu
        .phys_device
        .surface_formats(&surface, Default::default())?
        .into_iter()
        .filter(|(format, _)| {
            *format == Format::R8G8B8A8_UNORM || *format == Format::B8G8R8A8_UNORM
        })
        .collect::<Vec<_>>()
        .remove(0);

    Ok((surface, surface_format))
}

fn init_app() -> R<App> {
    let (conn, screen_index) = xcb::Connection::connect(None)?;
    let setup = conn.get_setup();
    let screen = setup.roots().nth(screen_index as usize).unwrap();
    let window = conn.generate_id();

    conn.check_request(conn.send_request_checked(&x::CreateWindow {
        depth: x::COPY_FROM_PARENT as u8,
        wid: window,
        parent: screen.root(),
        x: 0,
        y: 0,
        width: 800,
        height: 600,
        border_width: 0,
        class: x::WindowClass::InputOutput,
        visual: screen.root_visual(),
        value_list: &[x::Cw::EventMask(
            x::EventMask::EXPOSURE | x::EventMask::KEY_PRESS | x::EventMask::STRUCTURE_NOTIFY,
        )],
    }))?;

    conn.check_request(conn.send_request_checked(&x::ChangeProperty {
        mode: x::PropMode::Replace,
        window,
        property: x::ATOM_WM_NAME,
        r#type: x::ATOM_STRING,
        data: "これは窓かな🙃".as_bytes(),
    }))?;

    let gpu = cached_result!(|| Gpu::new());
    let conn = ConstNode::new(Arc::new(conn));
    let window = ConstNode::new(window);
    let surface_and_format = cached_result!(|gpu, conn, window| init_surface(gpu?, &conn, &window));
    let surface = surface_and_format.map(|sf| sf.0);
    let surface_format = surface_and_format.map(|sf| sf.1);

    let janitor = Janitor::new();
    let simple_pass = cached_result!(|gpu, surface_format| RenderPass::new(
        gpu?,
        SimplePass {
            format: surface_format?
        }
    ));
    let player_pipeline = cached_result!(|gpu, simple_pass| GraphicsPipeline::new(
        gpu?,
        pipelines::player::Pipeline {},
        &simple_pass?
    ));

    Ok(App {
        running: true,
        conn,
        window: Window {
            handle: window,
            surface,
            surface_format,
            frame: None,
            extent: ConstNode::new(None),
        },
        gpu,
        janitor,
        simple_pass,
        player_pipeline,
        present_future: None,
        needs_render: false,
        num_samples: 1,
        position: Vector2f::new([0.0, 0.0]),
        movements: VecDeque::new(),
    })
}

#[derive(Debug)]
struct RenderError {
    cause: Box<dyn Error + 'static>,
}

impl RenderError {
    fn new(cause: Box<dyn Error + 'static>) -> Self {
        RenderError { cause }
    }
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rendering error")
    }
}

impl Error for RenderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.cause.borrow())
    }
}

#[derive(Debug)]
enum Event {
    Timeout,
    GuiError(xcb::Error),
    GuiEvent(Vec<xcb::Event>),
}

impl poll::Port<Event> for xcb::Connection {
    fn port_fd(&self) -> libc::c_int {
        self.as_raw_fd()
    }

    fn get_event(&self) -> Option<Event> {
        match try {
            let mut events = Vec::new();
            while let Some(event) = self.poll_for_event()? {
                events.push(event)
            }
            events
        } {
            Ok(events) => {
                if events.is_empty() {
                    None
                } else {
                    Some(Event::GuiEvent(events))
                }
            }
            Err(err) => Some(Event::GuiError(err)),
        }
    }
}

trait AppEvent {
    fn handle_event(self, app: &mut App) -> R<()>;
}

impl AppEvent for x::KeyPressEvent {
    fn handle_event(self, app: &mut App) -> R<()> {
        let code = self.detail();
        match code {
            0x09 => {
                app.running = false;
            }
            0x6F | 0x71 | 0x74 | 0x72 => {
                let mut now = Instant::now() - Duration::from_millis(12);
                let fractions = 22;
                let dist = 0.05;
                let dir = match code {
                    0x6F => Vector2f::new([0.0, -dist / fractions as f32]),
                    0x71 => Vector2f::new([-dist / fractions as f32, 0.0]),
                    0x74 => Vector2f::new([0.0, dist / fractions as f32]),
                    0x72 => Vector2f::new([dist / fractions as f32, 0.0]),
                    _ => unreachable!(),
                };
                for _ in 0..fractions {
                    app.movements.push_back((now, dir));
                    now += Duration::from_millis(84 / fractions);
                }
            }
            0x0A => {
                app.needs_render = true;
                app.num_samples = 1
            }
            0x0B => {
                app.needs_render = true;
                app.num_samples = 2
            }
            0x0C => {
                app.needs_render = true;
                app.num_samples = 4
            }
            0x0D => {
                app.needs_render = true;
                app.num_samples = 8
            }
            _ => println!("Key: {code:#04x} {:?}", self.time()),
        }

        Ok(())
    }
}

impl AppEvent for x::ConfigureNotifyEvent {
    fn handle_event(self, app: &mut App) -> R<()> {
        app.needs_render = true;
        if self.window() == app.window.handle.get() {
            app.window
                .extent
                .put(Some([self.width() as u32, self.height() as u32]));
        }
        Ok(())
    }
}

impl AppEvent for x::ExposeEvent {
    fn handle_event(self, app: &mut App) -> R<()> {
        app.needs_render = true;
        Ok(())
    }
}

impl AppEvent for xcb::Event {
    fn handle_event(self, app: &mut App) -> R<()> {
        match self {
            xcb::Event::X(x::Event::KeyPress(ev)) => app.handle_event(ev),
            xcb::Event::X(x::Event::ConfigureNotify(ev)) => app.handle_event(ev),
            xcb::Event::X(x::Event::Expose(ev)) => app.handle_event(ev),
            _ => {
                dont! { println!("Event: {self:?}"); };
                Ok(())
            }
        }
    }
}

impl AppEvent for Event {
    fn handle_event(self, app: &mut App) -> R<()> {
        match self {
            Event::Timeout => {
                let now = Instant::now();
                while let Some((inst, dir)) = app.movements.pop_front() {
                    if inst < now {
                        dont! { print!("move {dir:?} "); }
                        app.position = app.position + dir;
                        app.needs_render = true;
                    } else {
                        dont! { println!("next {:?}", inst.duration_since(now)); }
                        app.movements.push_front((inst, dir));
                        break;
                    }
                }
                if app.movements.is_empty() {
                    dont! { println!("done"); }
                }
            }
            Event::GuiError(err) => {
                Err(err)?;
            }
            Event::GuiEvent(gui_events) => {
                for gui_event in gui_events {
                    app.handle_event(gui_event)?;
                }
            }
        }
        Ok(())
    }
}

impl App {
    fn show_window(&self) -> R<()> {
        let conn = self.conn.get();
        Ok(conn.check_request(conn.send_request_checked(&x::MapWindow {
            window: self.window.handle.get(),
        }))?)
    }

    fn create_frame(&self) -> R<Frame> {
        let gpu = self.gpu.get()?;
        let surface = self.window.surface.get()?;
        let caps = gpu
            .phys_device
            .surface_capabilities(&surface, Default::default())?;
        let (chain, images) = Swapchain::new(
            gpu.device.clone(),
            self.window.surface.get()?.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format: Some(self.window.surface_format.get()?),
                image_usage: caps.supported_usage_flags,
                present_mode: PresentMode::Mailbox,
                ..Default::default()
            },
        )?;
        Ok(Frame { chain, images })
    }

    fn recreate_frame(&self, frame: &Frame) -> R<Frame> {
        let gpu = self.gpu.get()?;
        let surface = self.window.surface.get()?;
        let caps = gpu
            .phys_device
            .surface_capabilities(&surface, Default::default())?;
        let (chain, images) = frame.chain.recreate(SwapchainCreateInfo {
            min_image_count: caps.min_image_count,
            image_usage: caps.supported_usage_flags,
            present_mode: PresentMode::Mailbox,
            ..Default::default()
        })?;
        Ok(Frame { chain, images })
    }

    fn ensure_frame(&mut self, extent: [u32; 2]) -> R<Frame> {
        match self.window.frame {
            None => {
                let frame = self.create_frame()?;
                self.window.frame = Some(frame.clone());
                Ok(frame)
            }
            Some(ref frame) => {
                if frame.chain.image_extent() == extent {
                    Ok(frame.clone())
                } else {
                    let frame = self.recreate_frame(&frame)?;
                    self.window.frame = Some(frame.clone());
                    Ok(frame)
                }
            }
        }
    }

    fn render(&mut self) -> R<()> {
        let mut watch: Stopwatch<&'static str> = Stopwatch::new();
        let gpu = self.gpu.get()?;

        let extent = match self.window.extent.get() {
            Some(ext) => ext,
            None => return Ok(()),
        };

        let viewport = Vector2f::new([extent[0] as f32, extent[1] as f32]);

        let frame = self.ensure_frame(extent)?;
        watch.tick("ensure frame");
        let render_pass = self.simple_pass.get()?;
        watch.tick("ensure render pass");
        let pipeline = self.player_pipeline.get()?;
        watch.tick("ensure pipeline");

        let (image_index, _, image_future) =
            swapchain::acquire_next_image(frame.chain.clone(), None)?;
        let image = frame.images[image_index].clone();
        watch.tick("acquire");

        let image_view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                format: Some(image.format()),
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects {
                        color: true,
                        ..ImageAspects::empty()
                    },
                    mip_levels: 0..1,
                    array_layers: 0..1,
                },
                ..Default::default()
            },
        )?;
        watch.tick("image view");

        let framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![image_view],
                ..Default::default()
            },
        )?;
        watch.tick("framebuffer");

        let stretch = if viewport.x() > viewport.y() {
            Vector2f::new([viewport.y() / viewport.x(), 1.0])
        } else {
            Vector2f::new([1.0, viewport.x() / viewport.y()])
        };
        let circle_radius = 0.08;
        let model_view = Matrix3f::scaling(stretch)
            * Matrix3f::translation(self.position)
            * Matrix3f::uniform_scaling(circle_radius)
            * Matrix3f::identity();
        let world = shaders::triangle::ty::World {
            viewport: [extent[0] as f32, extent[1] as f32],
            pos: self.position.data(),
            modelView: Matrix4f::embed_matrix3(model_view.transpose()).data(),
            numSamples: self.num_samples,
        };
        let world_buffer = gpu.uniform_buffer_pool.from_data(world)?;
        watch.tick("world buffer");

        let layout = pipeline.handle().layout().set_layouts()[0].clone();
        let set = PersistentDescriptorSet::new(
            layout,
            [WriteDescriptorSet::buffer(0, world_buffer.clone())],
        )?;
        watch.tick("descriptor set");

        let mut cb = AutoCommandBufferBuilder::primary(
            gpu.device.clone(),
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        cb.set_viewport(
            0,
            Some(Viewport {
                origin: [0.0, 0.0],
                dimensions: [extent[0] as f32, extent[1] as f32],
                depth_range: 0.0..1.0,
            }),
        );
        cb.set_scissor(
            0,
            Some(Scissor {
                origin: [0, 0],
                dimensions: extent,
            }),
        );
        let active_render_pass = framebuffer.begin_render_pass(&mut cb, [0.05, 0.03, 0.12, 1.0])?;
        cb.bind_pipeline_graphics(pipeline.handle().clone());
        cb.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.handle().layout().clone(),
            0,
            set,
        );
        cb.draw(4, 1, 0, 0)?;
        framebuffer.end_render_pass(&mut cb, active_render_pass)?;

        let cmd = cb.build()?;
        watch.tick("build command buffer");

        let present_future = match std::mem::replace(&mut self.present_future, None) {
            None => image_future.boxed_send(),
            Some(mut last_future) => {
                last_future.cleanup_finished();
                last_future.join(image_future).boxed_send()
            }
        };
        watch.tick("cleanup");

        self.present_future = Some(Arc::new(
            present_future
                .then_execute(gpu.queue.clone(), cmd)?
                .then_swapchain_present(
                    gpu.queue.clone(),
                    PresentInfo {
                        index: image_index,
                        ..PresentInfo::swapchain(frame.chain.clone())
                    },
                )
                .boxed_send()
                .then_signal_fence_and_flush()?,
        ));
        watch.tick("flush");

        dont! {
            let periods = watch.periods();
            println!("{:?}", periods);
            let total: Duration = periods.into_iter().map(|(_, dur)| dur).sum();
            println!("Total: {total:?}");
        }
        Ok(())
    }

    fn handle_event(&mut self, event: impl AppEvent) -> R<()> {
        event.handle_event(self)
    }

    fn event_loop(mut self) -> R<()> {
        let mut last_action = Instant::now();
        while self.running {
            let conn = self.conn.get();
            let ports: [&dyn poll::Port<Event>; 1] = [&*conn];
            let poll_inst = Instant::now();
            let timeout = self
                .movements
                .front()
                .map(|(inst, _)| (inst.duration_since(poll_inst), Event::Timeout));
            let events = poll::wait_for_event(ports, timeout);
            let events_inst = Instant::now();
            dont! { println!("action {:?} / poll {:?}", poll_inst.duration_since(last_action), events_inst.duration_since(poll_inst)); }
            last_action = events_inst;

            for event in events {
                self.handle_event(event)?;
            }

            if self.needs_render {
                self.render().map_err(RenderError::new)?;
                self.needs_render = false;
            }
        }

        Ok(())
    }
}

pub fn run() -> R<()> {
    let app = init_app()?;
    dont! { println!("Init done"); }
    dont! { app.janitor.dispose(42)?; }
    app.show_window()?;
    app.event_loop()
}
