pub mod passes {
    use std::sync::Arc;
    use vulkano::{
        format::{ClearValue, Format},
        image::ImageLayout,
        render_pass::{
            AttachmentDescription, AttachmentReference, LoadOp, RenderPass as VkRenderPass,
            RenderPassCreateInfo, StoreOp, SubpassDescription,
        },
    };

    use crate::gpu::Gpu;
    use crate::prelude::*;
    use crate::render::RenderPassSpec;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct SimplePass {
        pub format: Format,
    }

    impl RenderPassSpec for SimplePass {
        type ClearValues = [f32; 4];

        fn create_render_pass(&self, gpu: Gpu) -> R<Arc<VkRenderPass>> {
            Ok(VkRenderPass::new(
                gpu.device.clone(),
                RenderPassCreateInfo {
                    attachments: vec![AttachmentDescription {
                        format: Some(self.format),
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Store,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::PresentSrc,
                        ..Default::default()
                    }],
                    subpasses: vec![SubpassDescription {
                        color_attachments: vec![Some(AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::General,
                            ..Default::default()
                        })],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )?)
        }

        fn make_clear_values(clear_values: Self::ClearValues) -> Vec<Option<ClearValue>> {
            vec![Some(ClearValue::Float(clear_values))]
        }
    }
}

pub mod pipelines {
    pub mod player {
        use std::sync::Arc;
        use vulkano::pipeline::{
            graphics::{
                color_blend::{
                    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
                },
                input_assembly::{InputAssemblyState, PrimitiveTopology},
                viewport::ViewportState,
            },
            GraphicsPipeline as VkGraphicsPipeline, PartialStateMode, StateMode,
        };

        use crate::gpu::Gpu;
        use crate::prelude::*;
        use crate::render::{GraphicsPipelineSpec, RenderPass};
        use crate::shaders::passes::SimplePass;

        vulkano_shaders::shader! {
            shaders: {
                vertex: { ty: "vertex", path: "triangle.vert" },
                fragment: { ty: "fragment", path: "triangle.frag" },
            },
            types_meta: {
                #[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
            },
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct Pipeline {}

        impl GraphicsPipelineSpec for Pipeline {
            type Pass = SimplePass;

            fn create_graphics_pipeline(
                &self,
                gpu: Gpu,
                pass: &RenderPass<SimplePass>,
            ) -> R<Arc<VkGraphicsPipeline>> {
                let pipeline = VkGraphicsPipeline::start()
                    .render_pass(pass.handle().first_subpass())
                    .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
                    .input_assembly_state(InputAssemblyState {
                        topology: PartialStateMode::Fixed(PrimitiveTopology::TriangleStrip),
                        ..Default::default()
                    })
                    .vertex_shader(
                        load_vertex(gpu.device.clone())?
                            .entry_point("main")
                            .ok_or("vertex entry point")?,
                        (),
                    )
                    .fragment_shader(
                        load_fragment(gpu.device.clone())?
                            .entry_point("main")
                            .ok_or("fragment entry point")?,
                        (),
                    )
                    .color_blend_state(ColorBlendState {
                        attachments: vec![ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_mask: ColorComponents::all(),
                            color_write_enable: StateMode::Fixed(true),
                        }],
                        ..Default::default()
                    })
                    .build(gpu.device.clone())?;
                Ok(pipeline)
            }
        }
    }
}

pub mod triangle {
    vulkano_shaders::shader! {
        shaders: {
            vertex: { ty: "vertex", path: "triangle.vert" },
            fragment: { ty: "fragment", path: "triangle.frag" },
        },
        types_meta: {
            #[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
        },
    }
}
