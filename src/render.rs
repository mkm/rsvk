use std::marker::PhantomData;
use std::sync::Arc;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    format::ClearValue,
    pipeline::GraphicsPipeline as VkGraphicsPipeline,
    render_pass::{
        Framebuffer as VkFramebuffer, FramebufferCreateInfo, RenderPass as VkRenderPass,
    },
};

use crate::gpu::Gpu;
use crate::prelude::*;

pub trait RenderPassSpec {
    type ClearValues;

    fn create_render_pass(&self, gpu: Gpu) -> R<Arc<VkRenderPass>>;
    fn make_clear_values(clear_values: Self::ClearValues) -> Vec<Option<ClearValue>>;
}

pub trait GraphicsPipelineSpec {
    type Pass: RenderPassSpec;

    fn create_graphics_pipeline(
        &self,
        gpu: Gpu,
        pass: &RenderPass<Self::Pass>,
    ) -> R<Arc<VkGraphicsPipeline>>;
}

#[derive(Debug, PartialEq, Eq)]
pub struct RenderPass<T: RenderPassSpec> {
    handle: Arc<VkRenderPass>,
    phantom: PhantomData<T>,
}

pub struct RenderPassLock<T: RenderPassSpec> {
    active: bool,
    phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct Framebuffer<T: RenderPassSpec> {
    handle: Arc<VkFramebuffer>,
    phantom: PhantomData<T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct GraphicsPipeline<T: GraphicsPipelineSpec> {
    handle: Arc<VkGraphicsPipeline>,
    phantom: PhantomData<T>,
}

impl<T: RenderPassSpec> Clone for RenderPass<T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            phantom: self.phantom,
        }
    }
}

impl<T: RenderPassSpec> RenderPass<T> {
    pub fn new(gpu: Gpu, info: T) -> R<Self> {
        Ok(Self {
            handle: info.create_render_pass(gpu)?,
            phantom: PhantomData,
        })
    }

    pub fn handle(&self) -> Arc<VkRenderPass> {
        self.handle.clone()
    }
}

impl<T: RenderPassSpec> Drop for RenderPassLock<T> {
    fn drop(&mut self) {
        if self.active {
            panic!("active render pass dropped");
        }
    }
}

impl<T: RenderPassSpec> Framebuffer<T> {
    pub fn new(pass: RenderPass<T>, info: FramebufferCreateInfo) -> R<Self> {
        Ok(Self {
            handle: VkFramebuffer::new(pass.handle.clone(), info)?,
            phantom: PhantomData,
        })
    }

    pub fn begin_render_pass(
        &self,
        cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        clear_values: T::ClearValues,
    ) -> R<RenderPassLock<T>> {
        cb.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: T::make_clear_values(clear_values),
                ..RenderPassBeginInfo::framebuffer(self.handle.clone())
            },
            SubpassContents::Inline,
        )?;
        Ok(RenderPassLock {
            active: true,
            phantom: PhantomData,
        })
    }

    pub fn end_render_pass(
        &self,
        cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        mut lock: RenderPassLock<T>,
    ) -> R<()> {
        cb.end_render_pass()?;
        lock.active = false;
        Ok(())
    }

    #[allow(unused)]
    pub fn handle(&self) -> Arc<VkFramebuffer> {
        self.handle.clone()
    }
}

impl<T: GraphicsPipelineSpec> Clone for GraphicsPipeline<T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            phantom: self.phantom,
        }
    }
}

impl<T: GraphicsPipelineSpec> GraphicsPipeline<T> {
    pub fn new(gpu: Gpu, info: T, pass: &RenderPass<T::Pass>) -> R<Self> {
        Ok(Self {
            handle: info.create_graphics_pipeline(gpu, pass)?,
            phantom: PhantomData,
        })
    }

    pub fn handle(&self) -> Arc<VkGraphicsPipeline> {
        self.handle.clone()
    }
}
