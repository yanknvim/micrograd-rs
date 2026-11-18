use micrograd_rs::{mlp::MLP, tape::{Tape, Var}};

fn main() {
    let dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    let mut model = MLP::with_seed(&[2, 4, 1], 42);
    let lr = 0.05;
    let epochs = 2000;

    let mut tape = Tape::new();

    for epoch in 0..=epochs {
        let mut total_loss = 0.0;

        for &(inputs, targets) in &dataset {
            tape.clear();

            let x: Vec<Var> = inputs.iter().map(|&v| tape.var(v)).collect();
            let (preds, vars) = model.forward(&tape, &x);

            let mut loss = tape.var(0.0);
            for (&pred, &t_val) in preds.iter().zip(&targets) {
                let diff = pred - tape.var(t_val);
                loss = loss + diff * diff;
            }
            total_loss += loss.val();

            let grads = tape.backward(loss);
            for (p, v) in model.params_mut().zip(vars) {
                *p -= lr * grads[v.id];
            }

        }

        if epoch % (epochs / 10) == 0 {
            println!("Epoch {:4} | Loss: {:.6}", epoch, total_loss / 4.0);
        }
    }

    println!("\n--- RESULT ---");
    for &(inputs, targets) in &dataset {
        tape.clear();
        let x: Vec<Var> = inputs.iter().map(|&v| tape.var(v)).collect();
        let (preds, _) = model.forward(&tape, &x);
        println!(
            "INPUT: ({:.0}, {:.0}) -> PRED: {:.4} (ANSWER: {:.0})",
            inputs[0],
            inputs[1],
            preds[0].val(),
            targets[0],
        )
    }
}

