use std::cell::RefCell;

/// Compact bit-set for tracking visited nodes during beam search.
/// Uses a generation counter to avoid O(n) clearing between searches.
pub struct VisitedSet {
    generations: Vec<u32>,
    current_gen: u32,
    capacity: usize,
}

impl VisitedSet {
    pub fn new(capacity: usize) -> Self {
        Self {
            generations: vec![0; capacity],
            current_gen: 1,
            capacity,
        }
    }

    /// Reset for a new search without zeroing the entire array.
    pub fn reset(&mut self, new_capacity: usize) {
        if new_capacity > self.capacity {
            self.generations.resize(new_capacity, 0);
            self.capacity = new_capacity;
        }
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.generations.fill(0);
            self.current_gen = 1;
        }
    }

    #[inline]
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < self.generations.len(), "VisitedSet::set out of bounds: {} >= {}", idx, self.generations.len());
        if idx < self.generations.len() {
            self.generations[idx] = self.current_gen;
        }
    }

    #[inline]
    pub fn contains(&self, idx: usize) -> bool {
        idx < self.generations.len() && self.generations[idx] == self.current_gen
    }

    pub fn clear(&mut self) {
        self.reset(self.capacity);
    }
}

thread_local! {
    static VISITED_POOL: RefCell<VisitedSet> = RefCell::new(VisitedSet::new(0));
}

/// Borrow a thread-local VisitedSet, resizing if necessary.
/// Avoids heap allocation on every search call.
pub fn with_visited<F, R>(capacity: usize, f: F) -> R
where
    F: FnOnce(&mut VisitedSet) -> R,
{
    VISITED_POOL.with(|pool| {
        let mut vs = pool.borrow_mut();
        vs.reset(capacity);
        f(&mut vs)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited_set() {
        let mut vs = VisitedSet::new(100);
        assert!(!vs.contains(0));
        assert!(!vs.contains(99));
        vs.set(42);
        assert!(vs.contains(42));
        assert!(!vs.contains(43));
        vs.set(0);
        vs.set(99);
        assert!(vs.contains(0));
        assert!(vs.contains(99));
        vs.clear();
        assert!(!vs.contains(42));
    }

    #[test]
    fn test_generation_reset() {
        let mut vs = VisitedSet::new(50);
        vs.set(10);
        assert!(vs.contains(10));
        vs.reset(50);
        assert!(!vs.contains(10));
        vs.set(20);
        assert!(vs.contains(20));
        assert!(!vs.contains(10));
    }

    #[test]
    fn test_pool_reuse() {
        let r1 = with_visited(100, |vs| {
            vs.set(42);
            assert!(vs.contains(42));
            42
        });
        assert_eq!(r1, 42);

        with_visited(100, |vs| {
            assert!(!vs.contains(42));
            vs.set(99);
            assert!(vs.contains(99));
        });
    }
}
