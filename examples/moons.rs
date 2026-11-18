use std::f32::consts::PI;

use rand::{RngExt, SeedableRng, rngs::StdRng};

use micrograd_rs::{
    mlp::MLP,
    tape::{Tape, Var},
};

fn make_moons(n_per_class: usize, noise: f32, seed: u64) -> Vec<([f32; 2], f32)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut dataset = Vec::with_capacity(n_per_class * 2);

    let mut jitter = || {
        // Sum of two uniform samples approximates a centered bell-like noise without extra crates.
        (rng.random_range(-noise..noise) + rng.random_range(-noise..noise)) * 0.5
    };

    for i in 0..n_per_class {
        let theta = PI * (i as f32) / (n_per_class.max(1) as f32);

        // Upper moon (label +1.0)
        dataset.push((
            [theta.cos() + jitter(), theta.sin() + jitter()],
            1.0,
        ));

        // Lower moon (label -1.0)
        dataset.push((
            [
                1.0 - theta.cos() + jitter(),
                1.0 - theta.sin() - 0.5 + jitter(),
            ],
            -1.0,
        ));
    }

    dataset
}

fn render_decision_boundary(model: &MLP, tape: &mut Tape, dataset: &[([f32; 2], f32)]) {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 24;
    const X_MIN: f32 = -1.4;
    const X_MAX: f32 = 2.4;
    const Y_MIN: f32 = -0.9;
    const Y_MAX: f32 = 1.4;

    let mut grid = vec![vec![' '; WIDTH]; HEIGHT];

    for (row_idx, row) in grid.iter_mut().enumerate() {
        // Flip y so higher values appear at the top of the terminal plot.
        let y = Y_MAX - (row_idx as f32 + 0.5) / (HEIGHT as f32) * (Y_MAX - Y_MIN);
        for (col_idx, cell) in row.iter_mut().enumerate() {
            let x = X_MIN + (col_idx as f32 + 0.5) / (WIDTH as f32) * (X_MAX - X_MIN);

            tape.clear();
            let input: Vec<Var> = [x, y].iter().map(|&v| tape.var(v)).collect();
            let (preds, _) = model.forward(tape, &input);
            let score = preds[0].val();

            *cell = if score.abs() < 0.15 {
                '*'
            } else if score >= 0.0 {
                '.'
            } else {
                ' '
            };
        }
    }

    for &(pt, label) in dataset {
        let col = ((pt[0] - X_MIN) / (X_MAX - X_MIN) * WIDTH as f32).floor() as isize;
        let row = ((Y_MAX - pt[1]) / (Y_MAX - Y_MIN) * HEIGHT as f32).floor() as isize;
        if col >= 0 && col < WIDTH as isize && row >= 0 && row < HEIGHT as isize {
            grid[row as usize][col as usize] = if label > 0.0 { 'O' } else { 'X' };
        }
    }

    println!("\n--- DECISION BOUNDARY ---");
    println!("+{}+", "-".repeat(WIDTH));
    for row in &grid {
        let line: String = row.iter().collect();
        println!("|{line}|");
    }
    println!("+{}+", "-".repeat(WIDTH));
    println!("Legend: O = class +1, X = class -1, . = pred >= 0, ' ' = pred < 0, * = boundary");
}

fn main() {
    const SEED: u64 = 42;
    let dataset = make_moons(50, 0.15, SEED);
    let n = dataset.len() as f32;

    let mut model = MLP::with_seed(&[2, 16, 16, 1], SEED);
    let num_params = model.params_mut().count();
    let mut grad_acc = vec![0.0_f32; num_params];

    let epochs = 300;
    let base_lr = 0.2;
    let weight_decay = 1e-4;

    let mut tape = Tape::new();

    for epoch in 0..=epochs {
        grad_acc.fill(0.0);
        let mut total_loss = 0.0;
        let mut correct = 0usize;

        for &(inputs, label) in &dataset {
            tape.clear();

            let x: Vec<Var> = inputs.iter().map(|&v| tape.var(v)).collect();
            let (preds, vars) = model.forward(&tape, &x);
            let pred = preds[0];

            // SVM hinge loss: max(0, 1 - y * pred)
            let loss = (tape.var(1.0) - tape.var(label) * pred).relu();
            total_loss += loss.val();

            if (pred.val() >= 0.0) == (label > 0.0) {
                correct += 1;
            }

            let grads = tape.backward(loss);
            for (acc, v) in grad_acc.iter_mut().zip(vars) {
                *acc += grads[v.id];
            }
        }

        let lr = base_lr * (1.0 - 0.9 * (epoch as f32 / epochs as f32));
        for (p, &g) in model.params_mut().zip(&grad_acc) {
            *p -= lr * (g / n + weight_decay * *p);
        }

        if epoch % (epochs / 10) == 0 {
            let acc = (correct as f32 / n) * 100.0;
            println!(
                "Epoch {:3} | Loss: {:.4} | Accuracy: {:5.1}%",
                epoch,
                total_loss / n,
                acc
            );
        }
    }

    render_decision_boundary(&model, &mut tape, &dataset);
}
