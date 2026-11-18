use std::cell::RefCell;
use std::ops::{Add, Sub, Mul, Neg};

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Var,
    Add { out: usize, lhs: usize, rhs: usize },
    Mul { out: usize, lhs: usize, rhs: usize },
    ReLU { out: usize, value: usize },
}

#[derive(Debug, Clone)]
struct TapeInner {
    values: Vec<f32>,
    ops: Vec<Op>,
}

/// Wengert list (tape) recording values and operations for reverse-mode autodiff.
#[derive(Debug, Clone)]
pub struct Tape {
    inner: RefCell<TapeInner>,
}

/// Lightweight handle to a scalar node stored in a `Tape`.
#[derive(Debug, Clone, Copy)]
pub struct Var<'a> {
    pub id: usize,
    pub tape: &'a Tape,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(TapeInner {
                values: Vec::new(),
                ops: Vec::new(),
            })
        }
    }

    pub fn clear(&mut self) {
        let inner = self.inner.get_mut();
        inner.values.clear();
        inner.ops.clear();
    }

    pub fn var(&self, val: f32) -> Var<'_> {
        let mut inner = self.inner.borrow_mut();
        let id = inner.values.len();
        inner.values.push(val);
        inner.ops.push(Op::Var);
        Var {
            id,
            tape: self
        }
    }

    pub fn backward(&self, root: Var) -> Vec<f32> {
        assert!(
            std::ptr::eq(self, root.tape),
            "root Var belongs to a different Tape"
        );
        let inner = self.inner.borrow();

        let mut grads = vec![0.0; inner.values.len()];

        grads[root.id] = 1.0;

        for op in inner.ops[..=root.id].iter().rev() {
            match *op {
                Op::Var => {},
                Op::Add { out, lhs, rhs } => {
                    let g = grads[out];
                    grads[lhs] += g;
                    grads[rhs] += g;
                }
                Op::Mul { out, lhs, rhs } => {
                    let g = grads[out];
                    grads[lhs] += inner.values[rhs] * g;
                    grads[rhs] += inner.values[lhs] * g;
                }
                Op::ReLU { out, value } => {
                    let g = grads[out];
                    if inner.values[out] > 0.0 {
                        grads[value] += g;
                    }
                }
            }
        }

        grads
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Var<'a> {
    pub fn relu(&self) -> Var<'a> {
        let mut inner = self.tape.inner.borrow_mut();
        let id = inner.values.len();

        let val = inner.values[self.id].max(0.0);

        inner.values.push(val);
        inner.ops.push(Op::ReLU { out: id, value: self.id });

        Var {
            id,
            tape: self.tape,
        }
    }

    pub fn val(&self) -> f32 {
        self.tape.inner.borrow().values[self.id]
    }

    pub fn backward(&self) -> Vec<f32> {
        self.tape.backward(*self)
    }
}

impl<'a> Add for Var<'a> {
    type Output = Var<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(
            std::ptr::eq(self.tape, rhs.tape),
            "cannot add Vars from different Tapes"
        );
        let mut inner = self.tape.inner.borrow_mut();
        let id = inner.values.len();

        let val = inner.values[self.id] + inner.values[rhs.id];

        inner.values.push(val);
        inner.ops.push(Op::Add { out: id, lhs: self.id, rhs: rhs.id });

        Var {
            id,
            tape: self.tape,
        }
    }
}

impl<'a> Mul for Var<'a> {
    type Output = Var<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(
            std::ptr::eq(self.tape, rhs.tape),
            "cannot multiply Vars from different Tapes"
        );
        let mut inner = self.tape.inner.borrow_mut();
        let id = inner.values.len();

        let val = inner.values[self.id] * inner.values[rhs.id];

        inner.values.push(val);
        inner.ops.push(Op::Mul { out: id, lhs: self.id, rhs: rhs.id });

        Var {
            id,
            tape: self.tape,
        }
    }
}

impl<'a> Neg for Var<'a> {
    type Output = Var<'a>;

    fn neg(self) -> Self::Output {
        self * self.tape.var(-1.0)
    }
}

impl<'a> Sub for Var<'a> {
    type Output = Var<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

