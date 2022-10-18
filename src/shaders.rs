pub mod triangle {
    vulkano_shaders::shader!{
        shaders: {
            vertex: { ty: "vertex", path: "triangle.vert" },
            fragment: { ty: "fragment", path: "triangle.frag" },
        },
        types_meta: {
            #[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
        },
    }
}
