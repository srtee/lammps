# Most Performant C++ Double Wrapper

This document describes the most performant approaches for a wrapper class to return either a static (defined once at runtime, though not at compile time) double, or a dynamic double (to be recalculated every time it is called).

The wrapper (or some interface of it) can be used as a straightforward double in subsequent code.

## 1. Zero-Overhead: `std::variant` with Branch (Recommended)

```cpp
class DoubleWrapper {
    std::variant<double, std::function<double()>> value;
    
public:
    // Static case - stores the double directly
    DoubleWrapper(double val) : value(val) {}
    
    // Dynamic case - stores the callable
    template<typename Func>
    DoubleWrapper(Func f) : value(std::function<double()>(f)) {}
    
    operator double() const {
        // Branch prediction works well if usage pattern is consistent
        if (std::holds_alternative<double>(value)) {
            return std::get<double>(value);
        }
        return std::get<std::function<double()>>(value)();
    }
};
```

**Why this is fast:**
- No heap allocation for the static case (double fits in variant storage)
- Single branch (predictable if you use mostly static or mostly dynamic)
- No vtable dispatch

## 2. Ultra-Lightweight: Union with Function Pointer

If you need absolute minimum overhead and can work with function pointers (no captures):

```cpp
class DoubleWrapper {
    union {
        double static_val;
        double (*dynamic_func)();
    };
    bool is_static;
    
public:
    DoubleWrapper(double val) : static_val(val), is_static(true) {}
    DoubleWrapper(double (*f)()) : dynamic_func(f), is_static(false) {}
    
    operator double() const {
        return is_static ? static_val : dynamic_func();
    }
};
```

**Pros:** No heap allocation, no type erasure overhead, just one branch  
**Cons:** Can't use lambdas with captures for the dynamic case

## 3. Compile-Time Polymorphism (CRTP)

If the choice is known at compile time, use CRTP for **zero runtime overhead**:

```cpp
template<typename Derived>
class DoubleWrapper {
public:
    operator double() const {
        return static_cast<const Derived*>(this)->get_value();
    }
};

class StaticDouble : public DoubleWrapper<StaticDouble> {
    double val;
public:
    explicit StaticDouble(double v) : val(v) {}
    double get_value() const { return val; }
};

class DynamicDouble : public DoubleWrapper<DynamicDouble> {
    std::function<double()> func;
public:
    explicit DynamicDouble(std::function<double()> f) : func(std::move(f)) {}
    double get_value() const { return func(); }
};

// Usage - no virtual dispatch, fully inlined
StaticDouble s(3.14);
double x = s;  // Compiles to: double x = 3.14;
```

## 4. `std::function_ref` (C++17/C++20 polyfill)

If you have access to C++20's `std::function_ref` (or implement a simple version):

```cpp
class DoubleWrapper {
    std::variant<double, std::function_ref<double()>> value;
public:
    operator double() const {
        return std::visit([](auto& v) -> double { return v; }, value);
    }
};
```

---

## Recommendation

**Use Option 1 (`std::variant`)** unless you're in an extremely hot path (called millions of times per second). It gives you:
- Type safety
- No heap allocation for static case
- Ability to use lambdas with captures
- Minimal overhead (one predictable branch)

**Use Option 3 (CRTP)** if you can determine static vs dynamic at compile time—it's genuinely zero-overhead.

**Avoid:** Virtual functions, `std::function` alone (heap allocation overhead), or `std::any` (type erasure overhead).

---

## Appendix: Lambdas with Captures

In **Option 2**, the limitation states: *"Can't use lambdas with captures for the dynamic case"*. Here's what that means:

### What is a Lambda?

A **lambda** is an anonymous (unnamed) function that you can define inline in your code. The basic syntax is:

```cpp
auto lambda = [](/* parameters */) { /* body */ };
```

### What are Captures?

The `[]` part is the **capture clause** — it allows the lambda to access variables from the surrounding scope:

| Syntax | Meaning |
|--------|---------|
| `[]` | Capture nothing (empty capture) |
| `[x]` | Capture `x` by value |
| `[&x]` | Capture `x` by reference |
| `[=]` | Capture all variables by value |
| `[&]` | Capture all variables by reference |

### Example of a Lambda WITH Captures

```cpp
int base = 10;

// Lambda that CAPTURES 'base' from the outer scope
auto addBase = [base](int x) { return base + x; };

// This lambda needs to store 'base' internally to work
double result = addBase(5);  // Returns 15
```

The lambda `addBase` **captures** the variable `base`, meaning it makes a copy of it to use later.

### Why Option 2 Can't Use Captures

Option 2 uses a **raw function pointer**:

```cpp
double (*dynamic_func)();  // Function pointer — NO capture storage
```

A raw function pointer can only point to:
- Regular functions: `double foo() { return 3.14; }`
- Lambdas **without** captures (stateless lambdas)

```cpp
// This lambda has NO captures — it CAN be used with Option 2
auto stateless = []() -> double { return 3.14; };

// This lambda HAS captures — it CANNOT be used with Option 2
int x = 42;
auto stateful = [x]() -> double { return x * 3.14; };
//           ^^^
//           Capture! This lambda carries state
```

Lambdas with captures need storage for their captured variables. Internally, the compiler transforms them into an **object with member variables**. A simple function pointer doesn't have space for this extra state.

### The Difference Illustrated

```cpp
// Option 2: Union with function pointer
// CAN use this:
double getPi() { return 3.14159; }
DoubleWrapper d1(getPi);

// CANNOT use this (needs capture):
int precision = 3;
auto getRoundedPi = [precision]() -> double { 
    return round(3.14159 * pow(10, precision)) / pow(10, precision); 
};
DoubleWrapper d2(getRoundedPi);  // ERROR! Can't convert to function pointer
```

### Summary

- **Captures** = variables from outer scope that a lambda "remembers"
- Option 2's function pointer approach is extremely lightweight but **stateless** — it can't store captured variables
- **Option 1** (`std::variant` with `std::function`) works with captures because `std::function` uses type erasure to store both the callable AND its captured state (though it may allocate heap memory for large captures)
