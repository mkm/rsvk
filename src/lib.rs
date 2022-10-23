#![feature(try_blocks)]

use std::error::Error;
use std::borrow::Borrow;
use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Instant, Duration};
use std::os::unix::io::AsRawFd;
use xcb::{x, Xid};
use vulkano::{
    VulkanLibrary,
    instance::{Instance, InstanceExtensions, InstanceCreateInfo, LayerProperties},
    device::{Device, DeviceExtensions, Features, DeviceCreateInfo, Queue, QueueCreateInfo, physical::PhysicalDevice},
    swapchain::{self, Swapchain, SwapchainCreateInfo, Surface, PresentMode, PresentInfo},
    image::{
        ImageLayout,
        ImageAccess,
        ImageAspects,
        ImageSubresourceRange,
        swapchain::SwapchainImage,
        view::{ImageView, ImageViewCreateInfo},
    },
    format::{ClearValue, Format},
    render_pass::{RenderPass, RenderPassCreateInfo, SubpassDescription, AttachmentDescription, AttachmentReference, LoadOp, StoreOp, Framebuffer, FramebufferCreateInfo},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents, RenderPassBeginInfo},
    pipeline::{
        Pipeline,
        PipelineBindPoint,
        StateMode,
        PartialStateMode,
        graphics::{
            GraphicsPipeline,
            viewport::{ViewportState, Viewport, Scissor},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            color_blend::{ColorBlendState, ColorBlendAttachmentState, ColorComponents, AttachmentBlend},
        },
    },
    sync::{GpuFuture, FenceSignalFuture},
    buffer::cpu_pool::CpuBufferPool,
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
};

#[macro_use]
mod prelude;
mod cached;
mod maths;
mod shaders;
mod stopwatch;
mod poll;
mod janitor;

use cached::Cached;

use maths::Vector2f;
use maths::Matrix3f;
use maths::Matrix4f;

use stopwatch::Stopwatch;

use janitor::Janitor;

type R<T> = Result<T, Box<dyn Error + 'static>>;

#[derive(Clone)]
struct Frame {
    chain: Arc<Swapchain<()>>,
    images: Vec<Arc<SwapchainImage<()>>>,
}

struct Window {
    handle: x::Window,
    surface: Arc<Surface<()>>,
    surface_format: Format,
    frame: Option<Frame>,
    extent: Option<[u32; 2]>,
}

#[derive(Clone, Eq)]
struct Gpu {
    phys_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    uniform_buffer_pool: Arc<CpuBufferPool<shaders::triangle::ty::World>>,
}

#[derive(Clone)]
struct TriangleRenderer {
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
}

type CreateTriangleRenderer = dyn FnMut(Gpu, Format) -> R<TriangleRenderer>;

struct App {
    running: bool,
    conn: xcb::Connection,
    window: Window,
    gpu: Gpu,
    janitor: Janitor,
    triangle_renderer: Box<CreateTriangleRenderer>,
    present_future: Option<Arc<FenceSignalFuture<Box<dyn GpuFuture + Send>>>>,
    needs_render: bool,
    num_samples: i32,
    position: Vector2f,
    movements: VecDeque<(Instant, Vector2f)>,
}

impl PartialEq for Gpu {
    fn eq(&self, that: &Self) -> bool {
        Arc::ptr_eq(&self.device, &that.device)
    }
}

fn want_layer(layer: &LayerProperties) -> bool {
    if cfg!(feature = "validation") && layer.name().contains("validation") {
        true
    } else if cfg!(feature = "api_dump") && layer.name().contains("api_dump") {
        true
    } else {
        false
    }
}

fn init_vulkan(conn: &xcb::Connection, window: &x::Window) -> R<(Gpu, Arc<Surface<()>>, Format)> {
    let lib = VulkanLibrary::new()?;

    dont!{
        let available_layers: Vec<_> =
            lib.layer_properties()?
            .map(|layer| format!("{}: {}", layer.name(), layer.description()))
            .collect();
        println!("Available layers: {available_layers:#?}");
    }

    let enabled_layers: Vec<_> =
        lib.layer_properties()?
        .filter(want_layer)
        .map(|layer| String::from(layer.name()))
        .collect();

    dont!{ println!("Enabled layers: {enabled_layers:#?}"); }

    let extensions = InstanceExtensions {
        khr_surface: true,
        khr_xcb_surface: true,
        khr_display: true,
        ext_display_surface_counter: true,
        .. InstanceExtensions::empty()
    };
    let vki = Instance::new(lib, InstanceCreateInfo {
        enabled_extensions: extensions,
        enabled_layers,
        .. Default::default()
    })?;

    let phys_device = vki.enumerate_physical_devices()?.next().ok_or("no physical device")?;
    println!("Device: {}", phys_device.properties().device_name);
    dont!{ println!("Timestamp period: {}", phys_device.properties().timestamp_period); }

    dont!{ println!("{:#?}", phys_device.supported_extensions()); }

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ext_display_control: true,
        .. DeviceExtensions::empty()
    };

    let (device, mut queues_iter) = Device::new(phys_device.clone(), DeviceCreateInfo {
        enabled_extensions: device_extensions,
        enabled_features: Features::empty(),
        queue_create_infos: vec![QueueCreateInfo {
            queue_family_index: 0,
            .. Default::default()
        }],
        .. Default::default()
    })?;
    let queue = queues_iter.next().ok_or("no queue")?;
    let uniform_buffer_pool = Arc::new(CpuBufferPool::uniform_buffer(device.clone()));

    let surface: Arc<Surface<()>> = unsafe { Surface::from_xcb(vki, conn.get_raw_conn(), window.resource_id(), ())? };

    let (surface_format, _surface_colour_space) =
        phys_device.surface_formats(&surface, Default::default())?.into_iter().filter(|(format, _)| {
            *format == Format::R8G8B8A8_UNORM || *format == Format::B8G8R8A8_UNORM
        }).collect::<Vec<_>>().remove(0);
    dont!{ println!("Surface Format: {surface_format:?}"); }

    Ok((Gpu {
        phys_device,
        device,
        queue,
        uniform_buffer_pool,
    }, surface, surface_format))
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
        value_list: &[
            x::Cw::EventMask(x::EventMask::EXPOSURE | x::EventMask::KEY_PRESS | x::EventMask::STRUCTURE_NOTIFY)
        ],
    }))?;

    conn.check_request(conn.send_request_checked(&x::ChangeProperty {
        mode: x::PropMode::Replace,
        window,
        property: x::ATOM_WM_NAME,
        r#type: x::ATOM_STRING,
        data: "これは窓かな🙃".as_bytes(),
    }))?;

    let (gpu, surface, surface_format) = init_vulkan(&conn, &window)?;

    let janitor = Janitor::new();
    let triangle_renderer =
        (Box::new(|gpu: Gpu, format: Format| {
            gpu.create_triangle_renderer(format)
        }) as Box<CreateTriangleRenderer>).cached();

    Ok(App {
        running: true,
        conn,
        window: Window {
            handle: window,
            surface,
            surface_format,
            frame: None,
            extent: None,
        },
        gpu,
        janitor,
        triangle_renderer,
        present_future: None,
        needs_render: false,
        num_samples: 1,
        position: Vector2f::new([0.0, 0.0]),
        movements: VecDeque::new(),
    })
}

impl Gpu {
    fn create_triangle_renderer(&self, format: Format) -> R<TriangleRenderer> {
        let render_pass = RenderPass::new(self.device.clone(), RenderPassCreateInfo {
            attachments: vec![AttachmentDescription {
                format: Some(format),
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::PresentSrc,
                .. Default::default()
            }],
            subpasses: vec![SubpassDescription {
                color_attachments: vec![Some(AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::General,
                    .. Default::default()
                })],
                .. Default::default()
            }],
            .. Default::default()
        })?;

        let pipeline = GraphicsPipeline::start()
            .render_pass(render_pass.clone().first_subpass())
            .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
            .input_assembly_state(InputAssemblyState {
                topology: PartialStateMode::Fixed(PrimitiveTopology::TriangleStrip),
                .. Default::default()
            })
            .vertex_shader(shaders::triangle::load_vertex(self.device.clone())?.entry_point("main").ok_or("vertex_entry_point")?, ())
            .fragment_shader(shaders::triangle::load_fragment(self.device.clone())?.entry_point("main").ok_or("fragment entry point")?, ())
            .color_blend_state(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: StateMode::Fixed(true),
                }],
                .. Default::default()
            })
            .build(self.device.clone())?;

        Ok(TriangleRenderer {
            render_pass,
            pipeline,
        })
    }
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
            },
            Err(err) => {
                Some(Event::GuiError(err))
            }
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
                    _ => unreachable!()
                };
                for _ in 0 .. fractions {
                    app.movements.push_back((now, dir));
                    now += Duration::from_millis(84 / fractions);
                }
            },
            0x0A => {
                app.needs_render = true;
                app.num_samples = 1
            },
            0x0B => {
                app.needs_render = true;
                app.num_samples = 2
            },
            0x0C => {
                app.needs_render = true;
                app.num_samples = 4
            },
            0x0D => {
                app.needs_render = true;
                app.num_samples = 8
            },
            _ =>
                println!("Key: {code:#04x} {:?}", self.time())
        }

        Ok(())
    }
}

impl AppEvent for x::ConfigureNotifyEvent {
    fn handle_event(self, app: &mut App) -> R<()> {
        app.needs_render = true;
        if self.window() == app.window.handle {
            app.window.extent = Some([self.width() as u32, self.height() as u32]);
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
            xcb::Event::X(x::Event::KeyPress(ev)) => {
                app.handle_event(ev)
            },
            xcb::Event::X(x::Event::ConfigureNotify(ev)) => {
                app.handle_event(ev)
            },
            xcb::Event::X(x::Event::Expose(ev)) => {
                app.handle_event(ev)
            },
            _ => {
                dont!{ println!("Event: {self:?}"); };
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
                        dont!{ print!("move {dir:?} "); }
                        app.position = app.position + dir;
                        app.needs_render = true;
                    } else {
                        dont!{ println!("next {:?}", inst.duration_since(now)); }
                        app.movements.push_front((inst, dir));
                        break;
                    }
                }
                if app.movements.is_empty() {
                    dont!{ println!("done"); }
                }
            },
            Event::GuiError(err) => {
                Err(err)?;
            },
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
        Ok(self.conn.check_request(self.conn.send_request_checked(&x::MapWindow {
            window: self.window.handle
        }))?)
    }

    fn create_frame(&self) -> R<Frame> {
        let caps = self.gpu.phys_device.surface_capabilities(&self.window.surface, Default::default())?;
        let (chain, images) = Swapchain::new(self.gpu.device.clone(), self.window.surface.clone(), SwapchainCreateInfo {
            min_image_count: caps.min_image_count,
            image_format: Some(self.window.surface_format),
            image_usage: caps.supported_usage_flags,
            present_mode: PresentMode::Mailbox,
            .. Default::default()
        })?;
        Ok(Frame {
            chain,
            images,
        })
    }

    fn recreate_frame(&self, frame: &Frame) -> R<Frame> {
        let caps = self.gpu.phys_device.surface_capabilities(&self.window.surface, Default::default())?;
        let (chain, images) = frame.chain.recreate(SwapchainCreateInfo {
            min_image_count: caps.min_image_count,
            image_usage: caps.supported_usage_flags,
            present_mode: PresentMode::Mailbox,
            .. Default::default()
        })?;
        Ok(Frame {
            chain,
            images,
        })
    }

    fn ensure_frame(&mut self, extent: [u32; 2]) -> R<Frame> {
        match self.window.frame {
            None => {
                let frame = self.create_frame()?;
                self.window.frame = Some(frame.clone());
                Ok(frame)
            },
            Some(ref frame) =>
                if frame.chain.image_extent() == extent {
                    Ok(frame.clone())
                } else {
                    let frame = self.recreate_frame(&frame)?;
                    self.window.frame = Some(frame.clone());
                    Ok(frame)
                }
        }
    }

    fn render(&mut self) -> R<()> {
        let mut watch: Stopwatch<&'static str> = Stopwatch::new();

        let extent = match self.window.extent {
            Some(ext) => ext,
            None => return Ok(())
        };

        let viewport = Vector2f::new([extent[0] as f32, extent[1] as f32]);

        let frame = self.ensure_frame(extent)?;
        watch.tick("ensure frame");
        let renderer = (self.triangle_renderer)(self.gpu.clone(), self.window.surface_format)?;
        watch.tick("ensure renderer");

        let (image_index, _, image_future) = swapchain::acquire_next_image(frame.chain.clone(), None)?;
        let image = frame.images[image_index].clone();
        watch.tick("acquire");

        let image_view = ImageView::new(image.clone(), ImageViewCreateInfo {
            format: Some(image.format()),
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects {
                    color: true,
                    .. ImageAspects::empty()
                },
                mip_levels: 0 .. 1,
                array_layers: 0 .. 1,
            },
            .. Default::default()
        })?;
        watch.tick("image view");

        let framebuffer = Framebuffer::new(renderer.render_pass.clone(), FramebufferCreateInfo {
            attachments: vec![image_view],
            .. Default::default()
        })?;
        watch.tick("framebuffer");

        let stretch = if viewport.x() > viewport.y() {
            Vector2f::new([viewport.y() / viewport.x(), 1.0])
        } else {
            Vector2f::new([1.0, viewport.x() / viewport.y()])
        };
        let circle_radius = 0.08;
        let model_view =
            Matrix3f::scaling(stretch) *
            Matrix3f::translation(self.position) *
            Matrix3f::uniform_scaling(circle_radius) *
            Matrix3f::identity();
        let world = shaders::triangle::ty::World {
            viewport: [extent[0] as f32, extent[1] as f32],
            pos: self.position.data(),
            modelView: Matrix4f::embed_matrix3(model_view.transpose()).data(),
            numSamples: self.num_samples,
        };
        let world_buffer = self.gpu.uniform_buffer_pool.from_data(world)?;
        watch.tick("world buffer");

        let layout = renderer.pipeline.layout().set_layouts()[0].clone();
        let set = PersistentDescriptorSet::new(layout, [WriteDescriptorSet::buffer(0, world_buffer.clone())])?;
        watch.tick("descriptor set");

        let mut cb = AutoCommandBufferBuilder::primary(
            self.gpu.device.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit)?;

        cb.set_viewport(0, Some(Viewport {
            origin: [0.0, 0.0],
            dimensions: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0 .. 1.0,
        }));
        cb.set_scissor(0, Some(Scissor {
            origin: [0, 0],
            dimensions: extent,
        }));
        cb.begin_render_pass(RenderPassBeginInfo {
            render_pass: renderer.render_pass,
            render_area_offset: [0, 0],
            render_area_extent: framebuffer.extent(),
            clear_values: vec![Some(ClearValue::Float([0.05, 0.03, 0.12, 1.0]))],
            .. RenderPassBeginInfo::framebuffer(framebuffer)
        }, SubpassContents::Inline)?;
        cb.bind_pipeline_graphics(renderer.pipeline.clone());
        cb.bind_descriptor_sets(PipelineBindPoint::Graphics, renderer.pipeline.layout().clone(), 0, set);
        cb.draw(4, 1, 0, 0)?;
        cb.end_render_pass()?;

        let cmd = cb.build()?;
        watch.tick("build command buffer");

        let present_future = match std::mem::replace(&mut self.present_future, None) {
            None => image_future.boxed_send(),
            Some(mut last_future) => {
                last_future.cleanup_finished();
                last_future.join(image_future).boxed_send()
            },
        };
        watch.tick("cleanup");

        self.present_future = Some(Arc::new(
            present_future
            .then_execute(self.gpu.queue.clone(), cmd)?
            .then_swapchain_present(self.gpu.queue.clone(), PresentInfo {
                index: image_index,
                .. PresentInfo::swapchain(frame.chain.clone())
            })
            .boxed_send()
            .then_signal_fence_and_flush()?
        ));
        watch.tick("flush");

        dont!{
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
            let ports: [&dyn poll::Port<Event>; 1] = [&self.conn];
            let poll_inst = Instant::now();
            let timeout = self.movements.front().map(|(inst, _)| (inst.duration_since(poll_inst), Event::Timeout));
            let events = poll::wait_for_event(ports, timeout);
            let events_inst = Instant::now();
            dont!{ println!("action {:?} / poll {:?}", poll_inst.duration_since(last_action), events_inst.duration_since(poll_inst)); }
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

pub fn run() -> R<()>{
    let app = init_app()?;
    dont!{ println!("Init done"); }
    dont!{ app.janitor.dispose(42)?; }
    app.show_window()?;
    app.event_loop()
}
