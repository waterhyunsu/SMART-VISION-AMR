// Optional exact raw-point geometry kernels. No ROS or NumPy C ABI dependency.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
bool native_double_format(const char* format) {
    return format && (std::strcmp(format, "d") == 0 ||
                      std::strcmp(format, "=d") == 0 ||
                      std::strcmp(format, "@d") == 0);
}
struct Matrix {
    Py_buffer view{};
    ~Matrix() { if (view.obj) PyBuffer_Release(&view); }
    bool get(PyObject* object, Py_ssize_t columns, const char* name) {
        if (PyObject_GetBuffer(object, &view, PyBUF_FORMAT | PyBUF_STRIDES) < 0)
            return false;
        if (view.ndim != 2 || !view.shape || view.shape[1] != columns ||
            view.itemsize != sizeof(double) || !native_double_format(view.format) ||
            !PyBuffer_IsContiguous(&view, 'C')) {
            PyErr_Format(PyExc_ValueError, "%s must be a C-contiguous float64 matrix with %zd columns", name, columns);
            return false;
        }
        const auto* values = static_cast<const unsigned char*>(view.buf);
        for (Py_ssize_t i = 0; i < view.len / static_cast<Py_ssize_t>(sizeof(double)); ++i) {
            double value;
            std::memcpy(&value, values + i*sizeof(double), sizeof(double));
            if (!std::isfinite(value)) {
                PyErr_Format(PyExc_ValueError, "%s contains a nonfinite value", name);
                return false;
            }
        }
        return true;
    }
    double at(Py_ssize_t row, Py_ssize_t column) const {
        double value;
        // memcpy also supports otherwise valid, unaligned NumPy views.
        std::memcpy(&value, static_cast<const char*>(view.buf) +
                    (row*view.shape[1]+column)*sizeof(double), sizeof(double));
        return value;
    }
};

double advance(double x, double y, double radius, double angle) {
    const double initial = radius > 0. ? -pi/2 : pi/2;
    const double bearing = std::atan2(y-radius, x);
    double result = std::fmod(angle > 0. ? bearing-initial : initial-bearing, 2*pi);
    if (result < 0.) result += 2*pi;
    return result;
}

double arc_distance(double x, double y, double r, double angle, double ex, double ey) {
    double result = std::min(std::hypot(x, y), std::hypot(x-ex, y-ey));
    if (std::abs(angle) >= 2*pi || advance(x, y, r, angle) <= std::abs(angle)+1.e-12)
        result = std::min(result, std::abs(std::hypot(x, y-r)-std::abs(r)));
    return result;
}

PyObject* output_buffer(Py_ssize_t count) {
    if (count < 0 || count > PY_SSIZE_T_MAX / static_cast<Py_ssize_t>(sizeof(double))) {
        PyErr_SetString(PyExc_OverflowError, "Geometry result is too large");
        return nullptr;
    }
    return PyBytes_FromStringAndSize(nullptr, count*sizeof(double));
}

PyObject* clearances(PyObject*, PyObject* args) {
    PyObject* object;
    double distance, angle, radius, longitudinal, lateral, yaw;
    if (!PyArg_ParseTuple(args, "Odddddd", &object, &distance, &angle, &radius,
                          &longitudinal, &lateral, &yaw)) return nullptr;
    if (!std::isfinite(distance) || !std::isfinite(angle) || !std::isfinite(radius) ||
        !std::isfinite(longitudinal) || !std::isfinite(lateral) || !std::isfinite(yaw) ||
        radius <= 0. || longitudinal < 0. || lateral < 0. || yaw < 0.) {
        PyErr_SetString(PyExc_ValueError, "Invalid raw clearance parameters");
        return nullptr;
    }
    Matrix points;
    if (!points.get(object, 3, "points")) return nullptr;
    const Py_ssize_t count = points.view.shape[0];
    PyObject* output = output_buffer(count);
    if (!output) return nullptr;
    char* destination = PyBytes_AS_STRING(output);
    const double correction = radius+lateral+2*std::abs(distance)*std::sin(std::min(pi, yaw)/2);
    const bool curved = distance != 0. && std::abs(angle) >= 1.e-6;
    const double r = curved ? distance/angle : 0.;
    const double ex = curved ? r*std::sin(angle) : 0.;
    const double ey = curved ? r*(1-std::cos(angle)) : 0.;
    if (!std::isfinite(correction) || !std::isfinite(r) || !std::isfinite(ex) ||
        !std::isfinite(ey) || (curved && !std::isfinite(r*r))) {
        Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError, "Raw clearance geometry overflow");
        return nullptr;
    }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < count; ++i) {
        const double x = points.at(i, 1), y = points.at(i, 2);
        const double lo = x-longitudinal, hi = x+longitudinal;
        const double gap = std::max(std::max(lo, -hi), 0.);
        double result;
        if (distance == 0.) {
            result = std::hypot(gap, y);
        } else if (!curved) {
            const double along = std::max(std::max(std::min(0., distance)-hi,
                                                   lo-std::max(0., distance)), 0.);
            result = std::hypot(along, y)-std::abs(distance*angle)*.5;
        } else {
            result = std::min(arc_distance(lo, y, r, angle, ex, ey),
                              arc_distance(hi, y, r, angle, ex, ey));
            result = std::min(result, std::hypot(gap, y));
            result = std::min(result, std::hypot(std::max(std::max(lo-ex, ex-hi), 0.), y-ey));
            if (std::abs(angle) >= pi)
                result = std::min(result, std::hypot(gap, y-2*r));
            const double discriminant = r*r-(y-r)*(y-r);
            if (discriminant >= 0.) {
                const double root = std::sqrt(discriminant);
                for (double value : {root, -root}) {
                    if (lo <= value && value <= hi && (std::abs(angle) >= 2*pi ||
                        advance(value, y, r, angle) <= std::abs(angle)+1.e-12))
                        result = 0.;
                }
            }
        }
        // Preserve Python's left-to-right subtraction order near hit thresholds.
        result = result-radius-lateral-2*std::abs(distance)*std::sin(std::min(pi, yaw)/2);
        std::memcpy(destination+i*sizeof(double), &result, sizeof(double));
    }
    Py_END_ALLOW_THREADS
    return output;
}

PyObject* box_clearances(PyObject*, PyObject* args) {
    PyObject *point_object, *box_object;
    double radius;
    if (!PyArg_ParseTuple(args, "OOd", &point_object, &box_object, &radius)) return nullptr;
    if (!std::isfinite(radius) || radius <= 0.) {
        PyErr_SetString(PyExc_ValueError, "Invalid box clearance radius");
        return nullptr;
    }
    Matrix points, boxes;
    if (!points.get(point_object, 3, "points")) return nullptr;
    // Public layout is (4, N): x-low, x-high, y-low, y-high.
    if (PyObject_GetBuffer(box_object, &boxes.view, PyBUF_FORMAT | PyBUF_STRIDES) < 0)
        return nullptr;
    if (boxes.view.ndim != 2 || !boxes.view.shape || boxes.view.shape[0] != 4 ||
        boxes.view.shape[1] < 1 || boxes.view.itemsize != sizeof(double) ||
        !native_double_format(boxes.view.format) ||
        !PyBuffer_IsContiguous(&boxes.view, 'C')) {
        PyErr_SetString(PyExc_ValueError, "boxes must be a C-contiguous float64 (4, N) matrix, N >= 1");
        return nullptr;
    }
    const Py_ssize_t box_count = boxes.view.shape[1];
    for (Py_ssize_t j = 0; j < box_count; ++j) {
        for (Py_ssize_t k = 0; k < 4; ++k) {
            if (!std::isfinite(boxes.at(k, j))) {
                PyErr_SetString(PyExc_ValueError, "Nonfinite box bound");
                return nullptr;
            }
        }
        if (boxes.at(0, j) > boxes.at(1, j) || boxes.at(2, j) > boxes.at(3, j)) {
            PyErr_SetString(PyExc_ValueError, "Reversed box bound");
            return nullptr;
        }
    }
    const Py_ssize_t count = points.view.shape[0];
    PyObject* output = output_buffer(count);
    if (!output) return nullptr;
    char* destination = PyBytes_AS_STRING(output);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < count; ++i) {
        const double x = points.at(i, 1), y = points.at(i, 2);
        double nearest = std::numeric_limits<double>::infinity();
        for (Py_ssize_t j = 0; j < box_count; ++j) {
            const double dx = std::max(std::max(boxes.at(0, j)-x, x-boxes.at(1, j)), 0.);
            const double dy = std::max(std::max(boxes.at(2, j)-y, y-boxes.at(3, j)), 0.);
            nearest = std::min(nearest, dx*dx+dy*dy);
        }
        const double clearance = std::sqrt(nearest)-radius;
        std::memcpy(destination+i*sizeof(double), &clearance, sizeof(double));
    }
    Py_END_ALLOW_THREADS
    return output;
}

PyMethodDef methods[] = {
    {"clearances", clearances, METH_VARARGS, "Exact directed-arc raw clearances as packed float64 bytes."},
    {"box_clearances", box_clearances, METH_VARARGS, "Exact raw-point distance to a union of closed boxes."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_geometry_native", nullptr, -1, methods};
}  // namespace

PyMODINIT_FUNC PyInit__geometry_native() { return PyModule_Create(&module); }
