use micrograd_rs::tape::Tape;

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual} (diff: {})",
        (actual - expected).abs()
    );
}

#[test]
fn basic_ops_and_relu_gradients() {
    let tape = Tape::new();
    let x = tape.var(2.0);
    let y = tape.var(-3.0);
    let z = tape.var(1.5);

    // f(x, y, z) = relu(x * y + z) - (x - z) * (-y)
    // With x=2, y=-3, z=1.5: x*y+z = -4.5 -> relu = 0
    // Second term: -(2 - 1.5) * 3 = -1.5
    let out = (x * y + z).relu() - (x - z) * (-y);
    assert_close(out.val(), -1.5, 1e-6);

    let grads = out.backward();
    // d/dx: 0 - 3 = -3
    // d/dy: 0 + (2 - 1.5) = 0.5
    // d/dz: 0 + 3 = 3
    assert_close(grads[x.id], -3.0, 1e-5);
    assert_close(grads[y.id], 0.5, 1e-5);
    assert_close(grads[z.id], 3.0, 1e-5);
}

#[test]
fn fanout_accumulates_gradients() {
    let tape = Tape::new();
    let x = tape.var(3.0);
    // f(x) = (x * x + x).relu() = x^2 + x  =>  f'(3) = 2*3 + 1 = 7
    let out = (x * x + x).relu();
    assert_close(out.val(), 12.0, 1e-6);

    let grads = out.backward();
    assert_close(grads[x.id], 7.0, 1e-5);
}

#[test]
fn backward_ignores_ops_after_root() {
    let tape = Tape::new();
    let x = tape.var(2.0);
    let root = x * x; // df/dx = 4.0
    let _later = root * tape.var(10.0) + x;

    let grads = tape.backward(root);
    assert_close(grads[x.id], 4.0, 1e-5);
}

#[test]
fn tape_clear_allows_reuse() {
    let mut tape = Tape::new();
    {
        let a = tape.var(2.0);
        let b = a * tape.var(5.0);
        assert_close(b.val(), 10.0, 1e-6);
    }

    tape.clear();
    let x = tape.var(4.0);
    let y = x + tape.var(1.0);
    assert_eq!(x.id, 0);
    assert_close(y.val(), 5.0, 1e-6);
    let grads = y.backward();
    assert_close(grads[x.id], 1.0, 1e-5);
}

#[test]
#[should_panic(expected = "different Tapes")]
fn add_vars_from_different_tapes_panics() {
    let t1 = Tape::new();
    let t2 = Tape::new();
    let _ = t1.var(1.0) + t2.var(2.0);
}

#[test]
#[should_panic(expected = "different Tapes")]
fn mul_vars_from_different_tapes_panics() {
    let t1 = Tape::new();
    let t2 = Tape::new();
    let _ = t1.var(1.0) * t2.var(2.0);
}

#[test]
#[should_panic(expected = "different Tape")]
fn backward_with_var_from_different_tape_panics() {
    let t1 = Tape::new();
    let t2 = Tape::new();
    let v = t2.var(1.0);
    let _ = t1.backward(v);
}
