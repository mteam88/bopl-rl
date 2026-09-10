use bopl_sim::fixed::{Fix, MathError};
use std::io::{self, BufRead, BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let parts: Vec<_> = line.split('\t').collect();
        if parts.len() != 3 {
            return Err("expected operation and two raw integers".into());
        }
        let a = Fix::from_raw(parts[1].parse()?);
        let b = Fix::from_raw(parts[2].parse()?);
        let result = match parts[0] {
            "add" => Ok(a + b),
            "sub" => Ok(a - b),
            "mul" => Ok(a * b),
            "div" => a.checked_div(b),
            "slow_add" => Ok(a.slow_add(b)),
            "slow_sub" => Ok(a.slow_sub(b)),
            "slow_mul" => Ok(a.slow_mul(b)),
            "rem" => a.checked_rem(b, false),
            "slow_rem" => a.checked_rem(b, true),
            "neg" => Ok(-a),
            "abs" => Ok(a.abs()),
            "slow_abs" => Ok(a.slow_abs()),
            "floor" => Ok(a.floor()),
            "ceil" => Ok(a.ceil()),
            "round" => Ok(a.round()),
            "sqrt" => a.sqrt(),
            "sin" => Ok(a.sin()),
            "cos" => Ok(a.cos()),
            "fast_sin" => Ok(a.fast_sin()),
            "fast_cos" => Ok(a.fast_cos()),
            "tan" => Ok(a.tan()),
            "acos" => a.acos(),
            "atan" => Ok(a.atan()),
            "atan2" => Ok(a.atan2(b)),
            _ => return Err(format!("unknown operation: {}", parts[0]).into()),
        };
        match result {
            Ok(value) => writeln!(output, "{}", value.raw())?,
            Err(MathError::DivisionByZero) => writeln!(output, "division_by_zero")?,
            Err(MathError::Domain) => writeln!(output, "domain")?,
            Err(MathError::Overflow) => writeln!(output, "overflow")?,
        }
    }
    Ok(())
}
