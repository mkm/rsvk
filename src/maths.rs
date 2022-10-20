use std::ops::{Add, Sub, Mul};

fn n<T : From<i8>>(v: i8) -> T {
    From::<i8>::from(v)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector<Data>(Data);

pub type Vector2f = Vector<[f32; 2]>;

impl<T : Clone> Vector<T> {
    pub fn new(data: T) -> Self {
        Vector(data)
    }

    pub fn data(&self) -> T {
        self.0.clone()
    }
}

impl<T : Clone> Vector<[T; 2]> {
    pub fn x(&self) -> T {
        self.0[0].clone()
    }

    pub fn y(&self) -> T {
        self.0[1].clone()
    }
}

impl<T : Add> Add for Vector<[T; 2]> {
    type Output = Vector<[<T as Add>::Output; 2]>;

    fn add(self, that: Self) -> Self::Output {
        let Vector([x1, y1]) = self;
        let Vector([x2, y2]) = that;
        Vector([x1 + x2, y1 + y2])
    }
}

impl<T : Add> Add for Vector<[T; 3]> {
    type Output = Vector<[<T as Add>::Output; 3]>;

    fn add(self, that: Self) -> Self::Output {
        let Vector([x1, y1, z1]) = self;
        let Vector([x2, y2, z2]) = that;
        Vector([x1 + x2, y1 + y2, z1 + z2])
    }
}

impl<T : Sub> Sub for Vector<[T; 3]> {
    type Output = Vector<[<T as Sub>::Output; 3]>;

    fn sub(self, that: Self) -> Self::Output {
        let Vector([x1, y1, z1]) = self;
        let Vector([x2, y2, z2]) = that;
        Vector([x1 - x2, y1 - y2, z1 - z2])
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Matrix<Data>(Data);

pub type Matrix3f = Matrix<[[f32; 3]; 3]>;
pub type Matrix4f = Matrix<[[f32; 4]; 4]>;

impl<T : Clone> Matrix<T> {
    pub fn data(&self) -> T {
        self.0.clone()
    }
}

impl<T : From<i8> + Clone> Matrix<[[T; 3]; 3]> {
    pub fn identity() -> Self {
        Matrix([
            [n(1), n(0), n(0)],
            [n(0), n(1), n(0)],
            [n(0), n(0), n(1)],
        ])
    }

    pub fn translation(offset: Vector<[T; 2]>) -> Self {
        Matrix([
            [n(1), n(0), offset.x()],
            [n(0), n(1), offset.y()],
            [n(0), n(0), n(1)      ],
        ])
    }

    pub fn scaling(scale: Vector<[T; 2]>) -> Self {
        Matrix([
            [scale.x(), n(0),      n(0)],
            [n(0),      scale.y(), n(0)],
            [n(0),      n(0),      n(1)],
        ])
    }

    pub fn uniform_scaling(scale: T) -> Self {
        Self::scaling(Vector::new([scale.clone(), scale]))
    }

    pub fn transpose(self) -> Self {
        let Matrix([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33],
        ]) = self;
        Matrix([
            [a11, a21, a31],
            [a12, a22, a32],
            [a13, a23, a33],
        ])
    }
}

impl<T : Copy + Add<Output = T> + Mul<Output = T>> Mul for Matrix<[[T; 3]; 3]> {
    type Output = Self;

    fn mul(self, that: Self) -> Self {
        let Matrix([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33],
        ]) = self;
        let Matrix([
            [b11, b12, b13],
            [b21, b22, b23],
            [b31, b32, b33],
        ]) = that;
        Matrix([
            [a11 * b11 + a12 * b21 + a13 * b31, a11 * b12 + a12 * b22 + a13 * b32, a11 * b13 + a12 * b23 + a13 * b33],
            [a21 * b11 + a22 * b21 + a23 * b31, a21 * b12 + a22 * b22 + a23 * b32, a21 * b13 + a22 * b23 + a23 * b33],
            [a31 * b11 + a32 * b21 + a33 * b31, a31 * b12 + a32 * b22 + a33 * b32, a31 * b13 + a32 * b23 + a33 * b33],
        ])
    }
}

impl<T : From<i8>> Matrix<[[T; 4]; 4]> {
    pub fn embed_matrix3(m: Matrix<[[T; 3]; 3]>) -> Self {
        let Matrix([
            [a11, a12, a14],
            [a21, a22, a24],
            [a41, a42, a44],
        ]) = m;
        Matrix([
            [a11,  a12,  n(0), a14 ],
            [a21,  a22,  n(0), a24 ],
            [n(0), n(0), n(1), n(0)],
            [a41,  a42,  n(0), a44 ],
        ])
    }
}
