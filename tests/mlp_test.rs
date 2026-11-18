use micrograd_rs::{
    mlp::{Linear, MLP},
    tape::Tape,
};

#[test]
fn linear_params_match_numerical_gradients() {
    let mut layer = Linear::with_seed(3, 2, 123);
    let inputs = [0.7_f32, -1.2, 0.4];
    let mut tape = Tape::new();

    let eval = |layer: &Linear, tape: &mut Tape| -> f32 {
        tape.clear();
        let x: Vec<_> = inputs.iter().map(|&v| tape.var(v)).collect();
        let pass = layer.forward(tape, &x);
        // Combine both outputs non-linearly so every weight and bias contributes.
        let loss = pass.out[0] * pass.out[0] + pass.out[1] * tape.var(0.5);
        loss.val()
    };

    let x: Vec<_> = inputs.iter().map(|&v| tape.var(v)).collect();
    let pass = layer.forward(&tape, &x);
    let loss = pass.out[0] * pass.out[0] + pass.out[1] * tape.var(0.5);
    let grads = tape.backward(loss);
    let analytic: Vec<f32> = pass.params.iter().map(|v| grads[v.id]).collect();

    let num_params = layer.params_mut().count();
    assert_eq!(analytic.len(), num_params);

    let eps = 1e-3_f32;
    for (idx, &grad) in analytic.iter().enumerate() {
        let orig = *layer.params_mut().nth(idx).unwrap();

        *layer.params_mut().nth(idx).unwrap() = orig + eps;
        let plus = eval(&layer, &mut tape);

        *layer.params_mut().nth(idx).unwrap() = orig - eps;
        let minus = eval(&layer, &mut tape);

        *layer.params_mut().nth(idx).unwrap() = orig;

        let numeric = (plus - minus) / (2.0 * eps);
        assert!(
            (grad - numeric).abs() <= 5e-3,
            "param {idx}: analytic={grad}, numeric={numeric}"
        );
    }
}

#[test]
fn mlp_params_match_numerical_gradients() {
    let mut model = MLP::with_seed(&[2, 4, 1], 42);
    let inputs = [0.6_f32, -0.8];
    let target = 1.0_f32;
    let mut tape = Tape::new();

    let eval = |model: &MLP, tape: &mut Tape| -> f32 {
        tape.clear();
        let x: Vec<_> = inputs.iter().map(|&v| tape.var(v)).collect();
        let pass = model.forward(tape, &x);
        let diff = pass.out[0] - tape.var(target);
        (diff * diff).val()
    };

    let x: Vec<_> = inputs.iter().map(|&v| tape.var(v)).collect();
    let pass = model.forward(&tape, &x);
    let diff = pass.out[0] - tape.var(target);
    let loss = diff * diff;
    let grads = tape.backward(loss);
    let analytic: Vec<f32> = pass.params.iter().map(|v| grads[v.id]).collect();

    let num_params = model.params_mut().count();
    assert_eq!(analytic.len(), num_params);

    let eps = 1e-3_f32;
    for (idx, &grad) in analytic.iter().enumerate() {
        let orig = *model.params_mut().nth(idx).unwrap();

        *model.params_mut().nth(idx).unwrap() = orig + eps;
        let plus = eval(&model, &mut tape);

        *model.params_mut().nth(idx).unwrap() = orig - eps;
        let minus = eval(&model, &mut tape);

        *model.params_mut().nth(idx).unwrap() = orig;

        let numeric = (plus - minus) / (2.0 * eps);
        assert!(
            (grad - numeric).abs() <= 5e-3,
            "param {idx}: analytic={grad}, numeric={numeric}"
        );
    }
}
