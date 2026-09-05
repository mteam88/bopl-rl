//! ArraySortHelper<T>'s Mono introsort. Equal-key ordering is observable in
//! Updater initialization, so Rust's own unstable sort is not interchangeable.
type Result<T> = std::result::Result<T, &'static str>;
pub fn sort<T: Copy>(
    items: &mut [T],
    length: usize,
    compare: impl Fn(T, T) -> Result<i32>,
) -> Result<()> {
    if length < 2 {
        return Ok(());
    }
    let depth = 2 * (usize::BITS - length.leading_zeros());
    intro(items, 0, length as isize - 1, depth, &compare)
}
fn get<T: Copy>(a: &[T], i: isize) -> Result<T> {
    a.get(i as usize).copied().ok_or("bad_comparer")
}
fn swap_greater<T: Copy>(
    a: &mut [T],
    i: isize,
    j: isize,
    c: &impl Fn(T, T) -> Result<i32>,
) -> Result<()> {
    if i != j && c(get(a, i)?, get(a, j)?)? > 0 {
        a.swap(i as usize, j as usize);
    }
    Ok(())
}
fn intro<T: Copy>(
    a: &mut [T],
    lo: isize,
    mut hi: isize,
    mut depth: u32,
    c: &impl Fn(T, T) -> Result<i32>,
) -> Result<()> {
    while hi > lo {
        let length = hi - lo + 1;
        if length <= 16 {
            match length {
                2 => swap_greater(a, lo, hi, c)?,
                3 => {
                    swap_greater(a, lo, hi - 1, c)?;
                    swap_greater(a, lo, hi, c)?;
                    swap_greater(a, hi - 1, hi, c)?;
                }
                _ => {
                    for i in lo..hi {
                        let value = get(a, i + 1)?;
                        let mut j = i;
                        while j >= lo && c(value, get(a, j)?)? < 0 {
                            a[(j + 1) as usize] = get(a, j)?;
                            j -= 1;
                        }
                        a[(j + 1) as usize] = value;
                    }
                }
            }
            return Ok(());
        }
        if depth == 0 {
            for i in (1..=length / 2).rev() {
                down_heap(a, i, length, lo, c)?;
            }
            for i in (2..=length).rev() {
                a.swap(lo as usize, (lo + i - 1) as usize);
                down_heap(a, 1, i - 1, lo, c)?;
            }
            return Ok(());
        }
        depth -= 1;
        let middle = lo + (hi - lo) / 2;
        swap_greater(a, lo, middle, c)?;
        swap_greater(a, lo, hi, c)?;
        swap_greater(a, middle, hi, c)?;
        let pivot = get(a, middle)?;
        a.swap(middle as usize, (hi - 1) as usize);
        let mut left = lo;
        let mut right = hi - 1;
        while left < right {
            loop {
                left += 1;
                if c(get(a, left)?, pivot)? >= 0 {
                    break;
                }
            }
            loop {
                right -= 1;
                if c(pivot, get(a, right)?)? >= 0 {
                    break;
                }
            }
            if left >= right {
                break;
            }
            a.swap(left as usize, right as usize);
        }
        a.swap(left as usize, (hi - 1) as usize);
        intro(a, left + 1, hi, depth, c)?;
        hi = left - 1;
    }
    Ok(())
}
fn down_heap<T: Copy>(
    a: &mut [T],
    mut i: isize,
    n: isize,
    lo: isize,
    c: &impl Fn(T, T) -> Result<i32>,
) -> Result<()> {
    let value = get(a, lo + i - 1)?;
    while i <= n / 2 {
        let mut child = 2 * i;
        if child < n && c(get(a, lo + child - 1)?, get(a, lo + child)?)? < 0 {
            child += 1;
        }
        if c(value, get(a, lo + child - 1)?)? >= 0 {
            break;
        }
        a[(lo + i - 1) as usize] = get(a, lo + child - 1)?;
        i = child;
    }
    a[(lo + i - 1) as usize] = value;
    Ok(())
}
