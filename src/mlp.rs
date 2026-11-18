use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::tape::{Tape, Var};

#[derive(Debug)]
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_seed(in_features, out_features, rand::random())
    }

    pub fn with_seed(in_features: usize, out_features: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w = (0..out_features)
            .map(|_| {
                (0..in_features)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        Self {
            in_features,
            out_features,
            w,
            b: vec![0.0; out_features],
        }
    }

    pub fn forward<'a>(&self, tape: &'a Tape, x: &[Var<'a>]) -> (Vec<Var<'a>>, Vec<Var<'a>>) {
        assert_eq!(
            x.len(),
            self.in_features,
            "expected {} input features, got {}",
            self.in_features,
            x.len()
        );
        let mut out = Vec::with_capacity(self.b.len());
        let mut vars = Vec::new();

        for (w_row, &b_val) in self.w.iter().zip(&self.b) {
            let b_var = tape.var(b_val);
            vars.push(b_var);

            let mut acc = b_var;

            for (&w_val, &x_var) in w_row.iter().zip(x) {
                let w_var = tape.var(w_val);
                vars.push(w_var);
                acc = acc + w_var * x_var;
            }

            out.push(acc);
        }

        (out, vars)
    }

    pub fn params_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.b
            .iter_mut()
            .zip(self.w.iter_mut())
            .flat_map(|(b, w_row)| std::iter::once(b).chain(w_row.iter_mut()))
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Linear>,
}

impl MLP {
    pub fn new(dims: &[usize]) -> Self {
        Self::with_seed(dims, rand::random())
    }

    pub fn with_seed(dims: &[usize], seed: u64) -> Self {
        assert!(
            dims.len() >= 2,
            "MLP requires at least input and output dimensions"
        );
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, pair)| Linear::with_seed(pair[0], pair[1], seed.wrapping_add(i as u64)))
            .collect();

        Self { layers }
    }

    pub fn forward<'a>(&self, tape: &'a Tape, x: &[Var<'a>]) -> (Vec<Var<'a>>, Vec<Var<'a>>) {
        let mut current = x.to_vec();
        let mut all_vars = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let (next, vars) = layer.forward(tape, &current);
            all_vars.extend(vars);

            current = if i < self.layers.len() - 1 {
                next.into_iter().map(|v| v.relu()).collect()
            } else {
                next
            }
        }

        (current, all_vars)
    }

    pub fn params_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.layers.iter_mut().flat_map(|l| l.params_mut())
    }
}

