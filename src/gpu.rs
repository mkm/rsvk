use std::sync::Arc;
use vulkano::{
    buffer::CpuBufferPool,
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions, LayerProperties},
    VulkanLibrary,
};

use crate::prelude::*;
use crate::shaders;

#[derive(Clone)]
pub struct Gpu {
    pub phys_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub uniform_buffer_pool: Arc<CpuBufferPool<shaders::triangle::ty::World>>,
}

impl PartialEq for Gpu {
    fn eq(&self, that: &Self) -> bool {
        Arc::ptr_eq(&self.device, &that.device)
    }
}

impl Eq for Gpu {}

fn want_layer(layer: &LayerProperties) -> bool {
    if cfg!(feature = "validation") && layer.name().contains("validation") {
        true
    } else if cfg!(feature = "api_dump") && layer.name().contains("api_dump") {
        true
    } else {
        false
    }
}

impl Gpu {
    pub fn new() -> R<Self> {
        let lib = VulkanLibrary::new()?;

        let enabled_layers: Vec<_> = lib
            .layer_properties()?
            .filter(want_layer)
            .map(|layer| String::from(layer.name()))
            .collect();

        let extensions = InstanceExtensions {
            khr_surface: true,
            khr_xcb_surface: true,
            khr_display: true,
            ext_display_surface_counter: true,
            ..InstanceExtensions::empty()
        };
        let vki = Instance::new(
            lib,
            InstanceCreateInfo {
                enabled_extensions: extensions,
                enabled_layers,
                ..Default::default()
            },
        )?;

        let phys_device = vki
            .enumerate_physical_devices()?
            .next()
            .ok_or("no physical device")?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ext_display_control: true,
            ..DeviceExtensions::empty()
        };

        let (device, mut queues_iter) = Device::new(
            phys_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: Features::empty(),
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        let queue = queues_iter.next().ok_or("no queue")?;
        let uniform_buffer_pool = Arc::new(CpuBufferPool::uniform_buffer(device.clone()));

        Ok(Self {
            phys_device,
            device,
            queue,
            uniform_buffer_pool,
        })
    }
}
