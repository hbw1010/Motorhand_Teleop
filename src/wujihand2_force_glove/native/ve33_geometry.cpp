#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

constexpr const char* capsule_name = "wujihand2_force_glove.ve33_geometry";

struct Buffer {
    Py_buffer view{};
    bool acquired = false;

    ~Buffer() { if (acquired) PyBuffer_Release(&view); }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    template<class T> T get(Py_ssize_t index) const {
        T result;
        std::memcpy(&result, static_cast<const char*>(view.buf) + index * sizeof(T), sizeof(T));
        return result;
    }
    void put(Py_ssize_t index, double value) {
        std::memcpy(static_cast<char*>(view.buf) + index * sizeof(double), &value, sizeof(double));
    }
};

bool native_format(const char* format, char code) {
    if (!format) return false;
    if (*format == '@' || *format == '=') ++format;
    else if (*format == '<' || *format == '>' || *format == '!') {
        const std::uint16_t word = 1;
        const bool little = *reinterpret_cast<const unsigned char*>(&word) == 1;
        if ((*format == '<') != little) return false;
        ++format;
    }
    return format[0] == code && format[1] == '\0';
}

bool acquire(Buffer& buffer, PyObject* object, const char* name, char kind,
             int ndim, const Py_ssize_t* shape, bool writable = false) {
    const int flags = PyBUF_FORMAT | PyBUF_STRIDES | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, &buffer.view, flags) < 0) {
        if (!PyErr_ExceptionMatches(PyExc_MemoryError)) {
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError, "%s must expose a %sC-contiguous buffer", name,
                         writable ? "writable " : "");
        }
        return false;
    }
    buffer.acquired = true;
    const auto& v = buffer.view;
    const bool correct_type = kind == 'd'
        ? v.itemsize == 8 && native_format(v.format, 'd')
        : kind == 'i'
            ? v.itemsize == 8 && (native_format(v.format, 'q') || native_format(v.format, 'l'))
            : v.itemsize == 1 && native_format(v.format, 'B');
    if (!correct_type) {
        PyErr_Format(PyExc_TypeError, "%s has an incompatible dtype or byte order", name);
        return false;
    }
    if (v.ndim != ndim || !v.shape || !PyBuffer_IsContiguous(&v, 'C') ||
        (writable && v.readonly)) {
        PyErr_Format(PyExc_ValueError, "%s must have the required shape and C-contiguous layout", name);
        return false;
    }
    // Check the complete extent without overflowing, even for hostile buffer exporters.
    Py_ssize_t count = 1;
    for (int i = 0; i < ndim; ++i) {
        if (v.shape[i] < 0 || (shape && shape[i] >= 0 && v.shape[i] != shape[i])) {
            PyErr_Format(PyExc_ValueError, "%s has an invalid shape", name);
            return false;
        }
        if (v.shape[i] && count > PY_SSIZE_T_MAX / v.shape[i]) {
            PyErr_Format(PyExc_ValueError, "%s shape is too large", name);
            return false;
        }
        count *= v.shape[i];
    }
    if (count > PY_SSIZE_T_MAX / v.itemsize || v.len != count * v.itemsize ||
        (v.len && !v.buf)) {
        PyErr_Format(PyExc_ValueError, "%s has an invalid buffer extent", name);
        return false;
    }
    return true;
}

bool finite(const Buffer& b, const char* name) {
    for (Py_ssize_t i = 0; i < b.view.len / 8; ++i) {
        if (!std::isfinite(b.get<double>(i))) {
            PyErr_Format(PyExc_ValueError, "%s contains a non-finite value", name);
            return false;
        }
    }
    return true;
}

bool overlaps(const Buffer& a, const Buffer& b) {
    if (!a.view.len || !b.view.len) return false;
    const auto pa = reinterpret_cast<std::uintptr_t>(a.view.buf);
    const auto pb = reinterpret_cast<std::uintptr_t>(b.view.buf);
    // Subtraction, rather than end addresses, avoids address wraparound.
    return pa <= pb ? pb - pa < static_cast<std::uintptr_t>(a.view.len)
                    : pa - pb < static_cast<std::uintptr_t>(b.view.len);
}

struct Model {
    Py_ssize_t joints;
    Py_ssize_t state_size;
    std::vector<std::int64_t> parents;
    std::vector<std::int64_t> indices;
    std::vector<std::uint8_t> kinds;
    std::vector<double> origins;
    std::vector<double> axes;
    std::vector<double> key;
    std::vector<double> world;
    std::vector<double> world_axes;
    bool cached = false;

    Model(Py_ssize_t j, Py_ssize_t s)
        : joints(j), state_size(s), parents(j), indices(j), kinds(j),
          origins(j * 16), axes(j * 3), key(s), world(j * 16), world_axes(j * 3) {}
};

void destroy_model(PyObject* capsule) {
    auto* model = static_cast<Model*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (model) delete model;
    else PyErr_Clear();
}

Model* model_from(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, capsule_name)) {
        PyErr_SetString(PyExc_TypeError, "expected a ve33 geometry model capsule");
        return nullptr;
    }
    return static_cast<Model*>(PyCapsule_GetPointer(capsule, capsule_name));
}

void multiply(const double* a, const double* b, double* out) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double value = 0;
            for (int k = 0; k < 4; ++k) value += a[4 * r + k] * b[4 * k + c];
            out[4 * r + c] = value;
        }
}

bool forward(Model& m, const Buffer& state) {
    bool same = m.cached;
    for (Py_ssize_t i = 0; same && i < m.state_size; ++i)
        same = m.key[i] == state.get<double>(i);
    if (same) return true;
    m.cached = false;
    for (Py_ssize_t j = 0; j < m.joints; ++j) {
        const double* origin = m.origins.data() + 16 * j;
        const double* axis = m.axes.data() + 3 * j;
        const double q = m.indices[j] < 0 ? 0.0 : state.get<double>(m.indices[j]);
        double local[16];
        if (m.kinds[j] == 0) {
            std::memcpy(local, origin, sizeof(local));
        } else {
            double motion[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            if (m.kinds[j] == 2) {
                for (int k = 0; k < 3; ++k) motion[k * 4 + 3] = axis[k] * q;
            } else {
                const double x = axis[0], y = axis[1], z = axis[2];
                const double c = std::cos(q), s = std::sin(q), cc = 1.0 - c;
                motion[0] = c + x * x * cc;
                motion[1] = x * y * cc - z * s;
                motion[2] = x * z * cc + y * s;
                motion[4] = y * x * cc + z * s;
                motion[5] = c + y * y * cc;
                motion[6] = y * z * cc - x * s;
                motion[8] = z * x * cc - y * s;
                motion[9] = z * y * cc + x * s;
                motion[10] = c + z * z * cc;
            }
            multiply(origin, motion, local);
        }
        double* world = m.world.data() + 16 * j;
        if (m.parents[j] < 0) std::memcpy(world, local, sizeof(local));
        else multiply(m.world.data() + 16 * m.parents[j], local, world);
        for (int k = 0; k < 16; ++k) {
            if (!std::isfinite(world[k])) {
                PyErr_SetString(PyExc_ValueError, "state produces non-finite world frames");
                return false;
            }
        }
        // Match ClosureSolver: use the child frame, including its joint motion.
        for (int r = 0; r < 3; ++r) {
            double value = 0;
            for (int k = 0; k < 3; ++k) value += world[4 * r + k] * axis[k];
            m.world_axes[3 * j + r] = value;
        }
    }
    for (Py_ssize_t i = 0; i < m.state_size; ++i) m.key[i] = state.get<double>(i);
    m.cached = true;
    return true;
}

PyObject* create(PyObject*, PyObject* args) {
    PyObject *p, *k, *s, *o, *a, *size;
    if (!PyArg_ParseTuple(args, "OOOOOO:create", &p, &k, &s, &o, &a, &size)) return nullptr;
    if (!PyLong_Check(size) || PyBool_Check(size)) {
        PyErr_SetString(PyExc_TypeError, "state_size must be a positive integer");
        return nullptr;
    }
    const Py_ssize_t state_size = PyLong_AsSsize_t(size);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "state_size is out of range");
        return nullptr;
    }
    if (state_size <= 0 || state_size > PY_SSIZE_T_MAX / 8) {
        PyErr_SetString(PyExc_ValueError, "state_size must be positive and fit in a buffer");
        return nullptr;
    }
    Buffer parents, kinds, indices, origins, axes;
    if (!acquire(parents, p, "parents", 'i', 1, nullptr)) return nullptr;
    const Py_ssize_t j = parents.view.shape[0];
    if (j > PY_SSIZE_T_MAX / (16 * 8)) {
        PyErr_SetString(PyExc_ValueError, "too many joints");
        return nullptr;
    }
    const Py_ssize_t one[] = {j}, matrix[] = {j, 4, 4}, axis_shape[] = {j, 3};
    if (!acquire(kinds, k, "kinds", 'b', 1, one) ||
        !acquire(indices, s, "state_indices", 'i', 1, one) ||
        !acquire(origins, o, "origins", 'd', 3, matrix) ||
        !acquire(axes, a, "axes", 'd', 2, axis_shape) ||
        !finite(origins, "origins") || !finite(axes, "axes")) return nullptr;
    for (Py_ssize_t n = 0; n < j; ++n) {
        const auto parent = parents.get<std::int64_t>(n);
        const auto index = indices.get<std::int64_t>(n);
        if (parent < -1 || parent >= n || index < -1 || index >= state_size ||
            kinds.get<std::uint8_t>(n) > 2) {
            PyErr_SetString(PyExc_ValueError, "invalid parent, joint kind, or state index");
            return nullptr;
        }
        double squared_norm = 0;
        for (int c = 0; c < 3; ++c) {
            const double value = axes.get<double>(3 * n + c);
            squared_norm += value * value;
        }
        if (!std::isfinite(squared_norm) || std::abs(squared_norm - 1.0) > 1e-10) {
            PyErr_SetString(PyExc_ValueError, "axes must be normalized nonzero vectors");
            return nullptr;
        }
    }
    try {
        auto model = std::make_unique<Model>(j, state_size);
        for (Py_ssize_t n = 0; n < j; ++n) {
            model->parents[n] = parents.get<std::int64_t>(n);
            model->indices[n] = indices.get<std::int64_t>(n);
            model->kinds[n] = kinds.get<std::uint8_t>(n);
        }
        if (j) {
            std::memcpy(model->origins.data(), origins.view.buf, static_cast<std::size_t>(origins.view.len));
            std::memcpy(model->axes.data(), axes.view.buf, static_cast<std::size_t>(axes.view.len));
        }
        PyObject* capsule = PyCapsule_New(model.get(), capsule_name, destroy_model);
        if (!capsule) return nullptr;
        model.release();
        return capsule;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        PyErr_SetString(PyExc_ValueError, "model metadata is too large");
        return nullptr;
    }
}

bool valid_links(const Buffer& links, Py_ssize_t joints, bool root, const char* name) {
    for (Py_ssize_t i = 0; i < links.view.shape[0]; ++i) {
        const auto index = links.get<std::int64_t>(i);
        if (index < (root ? -1 : 0) || index >= joints) {
            PyErr_Format(PyExc_ValueError, "%s contains an out-of-range joint index", name);
            return false;
        }
    }
    return true;
}

void point(const Model& m, const Buffer& links, const Buffer& points, Py_ssize_t w, double* out) {
    const auto link = links.get<std::int64_t>(w);
    if (link < 0) {
        for (int r = 0; r < 3; ++r) out[r] = points.get<double>(4 * w + r);
        return;
    }
    const double* frame = m.world.data() + 16 * link;
    for (int r = 0; r < 3; ++r) {
        double value = 0;
        for (int k = 0; k < 4; ++k) value += frame[4 * r + k] * points.get<double>(4 * w + k);
        out[r] = value;
    }
}

void derivative(const Model& m, Py_ssize_t joint, const double* endpoint, double* out) {
    const double* axis = m.world_axes.data() + 3 * joint;
    if (m.kinds[joint] == 2) {
        for (int r = 0; r < 3; ++r) out[r] = axis[r];
    } else {
        // Fixed dependencies intentionally follow the original solver's cross product too.
        const double* frame = m.world.data() + 16 * joint;
        const double x = endpoint[0] - frame[3];
        const double y = endpoint[1] - frame[7];
        const double z = endpoint[2] - frame[11];
        out[0] = axis[1] * z - axis[2] * y;
        out[1] = axis[2] * x - axis[0] * z;
        out[2] = axis[0] * y - axis[1] * x;
    }
}

PyObject* evaluate(PyObject*, PyObject* args) {
    PyObject *capsule, *state_obj, *la, *pa, *lb, *pb, *dep, *ma, *mb, *res, *jac;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO:evaluate", &capsule, &state_obj,
                          &la, &pa, &lb, &pb, &dep, &ma, &mb, &res, &jac)) return nullptr;
    Model* m = model_from(capsule);
    if (!m) return nullptr;
    Buffer state, links_a, points_a, links_b, points_b, deps, mask_a, mask_b, residual, jacobian;
    const Py_ssize_t state_shape[] = {m->state_size};
    if (!acquire(state, state_obj, "state", 'd', 1, state_shape) ||
        !acquire(links_a, la, "links_a", 'i', 1, nullptr) ||
        !acquire(deps, dep, "deps", 'i', 1, nullptr)) return nullptr;
    const Py_ssize_t w = links_a.view.shape[0], d = deps.view.shape[0];
    if (w > PY_SSIZE_T_MAX / 3) {
        PyErr_SetString(PyExc_ValueError, "too many witnesses");
        return nullptr;
    }
    const Py_ssize_t link_shape[] = {w}, point_shape[] = {w, 4}, mask_shape[] = {w, d};
    const Py_ssize_t residual_shape[] = {3 * w}, jac_shape[] = {3 * w, d};
    if (!acquire(links_b, lb, "links_b", 'i', 1, link_shape) ||
        !acquire(points_a, pa, "points_a", 'd', 2, point_shape) ||
        !acquire(points_b, pb, "points_b", 'd', 2, point_shape) ||
        !acquire(mask_a, ma, "mask_a", 'b', 2, mask_shape) ||
        !acquire(mask_b, mb, "mask_b", 'b', 2, mask_shape) ||
        !acquire(residual, res, "out_residual", 'd', 1, residual_shape, true) ||
        !acquire(jacobian, jac, "out_jacobian", 'd', 2, jac_shape, true) ||
        !finite(state, "state") || !finite(points_a, "points_a") || !finite(points_b, "points_b") ||
        !valid_links(links_a, m->joints, true, "links_a") ||
        !valid_links(links_b, m->joints, true, "links_b") ||
        !valid_links(deps, m->joints, false, "deps")) return nullptr;
    for (Py_ssize_t i = 0; i < mask_a.view.len; ++i) {
        if (mask_a.get<std::uint8_t>(i) > 1 || mask_b.get<std::uint8_t>(i) > 1) {
            PyErr_SetString(PyExc_ValueError, "ancestor masks must contain only zero or one");
            return nullptr;
        }
    }
    const Buffer* inputs[] = {&state, &links_a, &points_a, &links_b, &points_b, &deps, &mask_a, &mask_b};
    for (const Buffer* input : inputs) {
        if (overlaps(residual, *input) || overlaps(jacobian, *input)) {
            PyErr_SetString(PyExc_ValueError, "output buffers must not overlap inputs");
            return nullptr;
        }
    }
    if (overlaps(residual, jacobian)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap each other");
        return nullptr;
    }
    if (!forward(*m, state)) return nullptr;
    // All validation is finished before any caller-owned output is touched.
    for (Py_ssize_t i = 0; i < w; ++i) {
        double a[3], b[3];
        point(*m, links_a, points_a, i, a);
        point(*m, links_b, points_b, i, b);
        for (int r = 0; r < 3; ++r) residual.put(3 * i + r, a[r] - b[r]);
        for (Py_ssize_t c = 0; c < d; ++c) {
            const Py_ssize_t joint = static_cast<Py_ssize_t>(deps.get<std::int64_t>(c));
            double da[3], db[3];
            derivative(*m, joint, a, da);
            derivative(*m, joint, b, db);
            const double am = mask_a.get<std::uint8_t>(i * d + c);
            const double bm = mask_b.get<std::uint8_t>(i * d + c);
            for (int r = 0; r < 3; ++r) jacobian.put((3 * i + r) * d + c, da[r] * am - db[r] * bm);
        }
    }
    Py_RETURN_NONE;
}

PyObject* frames(PyObject*, PyObject* args) {
    PyObject *capsule, *state_obj, *output_obj;
    if (!PyArg_ParseTuple(args, "OOO:frames", &capsule, &state_obj, &output_obj)) return nullptr;
    Model* m = model_from(capsule);
    if (!m) return nullptr;
    Buffer state, output;
    const Py_ssize_t state_shape[] = {m->state_size}, frame_shape[] = {m->joints, 4, 4};
    if (!acquire(state, state_obj, "state", 'd', 1, state_shape) ||
        !acquire(output, output_obj, "out_frames", 'd', 3, frame_shape, true) ||
        !finite(state, "state")) return nullptr;
    if (overlaps(state, output)) {
        PyErr_SetString(PyExc_ValueError, "out_frames must not overlap state");
        return nullptr;
    }
    if (!forward(*m, state)) return nullptr;
    if (output.view.len)
        std::memcpy(output.view.buf, m->world.data(), static_cast<std::size_t>(output.view.len));
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Create an owned, immutable topology and reusable FK workspace."},
    {"evaluate", evaluate, METH_VARARGS, "Write point-closure residuals and the original analytic Jacobian."},
    {"frames", frames, METH_VARARGS, "Write world frames in topology order."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_ve33_geometry",
    "Native vE3.3 FK and point-closure geometry; calls retain the GIL.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__ve33_geometry() {
    return PyModule_Create(&module);
}
