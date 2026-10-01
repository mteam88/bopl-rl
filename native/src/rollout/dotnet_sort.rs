//! .NET's `List<T>.Sort(IComparer<T>)`: ArraySortHelper<T>.IntrospectiveSort
//! (System.Private.CoreLib, ArraySortHelper.cs). It is not stable: equal
//! elements land where its insertion sort, median-of-three partition and
//! heapsort leave them, so a port that mirrors it orders every tie as C# does.

use std::cmp::Ordering;

const INTROSORT_SIZE_THRESHOLD: usize = 16;

/// Sorts `keys` exactly as `List<T>.Sort(comparer)` would.
pub(crate) fn introspective_sort<T: Copy>(keys: &mut [T], cmp: impl Fn(&T, &T) -> Ordering) {
    if keys.len() > 1 {
        let depth_limit = 2 * (keys.len().ilog2() as i32 + 1);
        intro_sort(keys, depth_limit, &cmp);
    }
}

fn intro_sort<T: Copy>(keys: &mut [T], mut depth_limit: i32, cmp: &impl Fn(&T, &T) -> Ordering) {
    let mut partition_size = keys.len();
    while partition_size > 1 {
        if partition_size <= INTROSORT_SIZE_THRESHOLD {
            if partition_size == 2 {
                swap_if_greater(keys, cmp, 0, 1);
                return;
            }
            if partition_size == 3 {
                swap_if_greater(keys, cmp, 0, 1);
                swap_if_greater(keys, cmp, 0, 2);
                swap_if_greater(keys, cmp, 1, 2);
                return;
            }
            insertion_sort(&mut keys[..partition_size], cmp);
            return;
        }
        if depth_limit == 0 {
            heap_sort(&mut keys[..partition_size], cmp);
            return;
        }
        depth_limit -= 1;
        let p = pick_pivot_and_partition(&mut keys[..partition_size], cmp);
        intro_sort(&mut keys[p + 1..partition_size], depth_limit, cmp);
        partition_size = p;
    }
}

fn swap_if_greater<T: Copy>(keys: &mut [T], cmp: &impl Fn(&T, &T) -> Ordering, i: usize, j: usize) {
    if cmp(&keys[i], &keys[j]) == Ordering::Greater {
        keys.swap(i, j);
    }
}

fn pick_pivot_and_partition<T: Copy>(keys: &mut [T], cmp: &impl Fn(&T, &T) -> Ordering) -> usize {
    let hi = keys.len() - 1;
    let middle = hi >> 1;
    swap_if_greater(keys, cmp, 0, middle);
    swap_if_greater(keys, cmp, 0, hi);
    swap_if_greater(keys, cmp, middle, hi);
    let pivot = keys[middle];
    keys.swap(middle, hi - 1);
    let (mut left, mut right) = (0, hi - 1);
    while left < right {
        left += 1;
        while cmp(&keys[left], &pivot) == Ordering::Less {
            left += 1;
        }
        right -= 1;
        while cmp(&pivot, &keys[right]) == Ordering::Less {
            right -= 1;
        }
        if left >= right {
            break;
        }
        keys.swap(left, right);
    }
    if left != hi - 1 {
        keys.swap(left, hi - 1);
    }
    left
}

fn heap_sort<T: Copy>(keys: &mut [T], cmp: &impl Fn(&T, &T) -> Ordering) {
    let n = keys.len();
    for i in (1..=n >> 1).rev() {
        down_heap(keys, i, n, cmp);
    }
    for i in (2..=n).rev() {
        keys.swap(0, i - 1);
        down_heap(keys, 1, i - 1, cmp);
    }
}

fn down_heap<T: Copy>(keys: &mut [T], mut i: usize, n: usize, cmp: &impl Fn(&T, &T) -> Ordering) {
    let d = keys[i - 1];
    while i <= n >> 1 {
        let mut child = 2 * i;
        if child < n && cmp(&keys[child - 1], &keys[child]) == Ordering::Less {
            child += 1;
        }
        if cmp(&d, &keys[child - 1]) != Ordering::Less {
            break;
        }
        keys[i - 1] = keys[child - 1];
        i = child;
    }
    keys[i - 1] = d;
}

fn insertion_sort<T: Copy>(keys: &mut [T], cmp: &impl Fn(&T, &T) -> Ordering) {
    for i in 0..keys.len() - 1 {
        let t = keys[i + 1];
        let mut j = i as isize;
        while j >= 0 && cmp(&t, &keys[j as usize]) == Ordering::Less {
            keys[j as usize + 1] = keys[j as usize];
            j -= 1;
        }
        keys[(j + 1) as usize] = t;
    }
}
