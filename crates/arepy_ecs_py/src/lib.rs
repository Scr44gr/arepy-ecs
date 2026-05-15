//! `PyO3` bindings that expose the Rust ECS core to Python.

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::sync::{Arc, Mutex, MutexGuard};

use arepy_ecs_core::{
    CoreError, FieldDefinition, FieldSnapshot, RuntimeValue, ValueKind, WorldCore,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use pyo3::IntoPyObject;

type SharedWorld = Arc<Mutex<WorldCore>>;

#[expect(
    clippy::needless_pass_by_value,
    reason = "`map_err(core_error)` passes `CoreError` by value at the FFI boundary"
)]
fn core_error(error: CoreError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn lock_error() -> PyErr {
    PyRuntimeError::new_err("failed to acquire ECS world lock")
}

fn lock_world(inner: &SharedWorld) -> PyResult<MutexGuard<'_, WorldCore>> {
    inner.lock().map_err(|_| lock_error())
}

fn py_ssize(value: usize, field: &str) -> PyResult<ffi::Py_ssize_t> {
    ffi::Py_ssize_t::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "{field} value `{value}` does not fit in Python's ssize_t"
        ))
    })
}

fn buffer_format(kind: ValueKind) -> [c_char; 2] {
    match kind {
        ValueKind::Bool => [b'?'.cast_signed(), 0],
        ValueKind::Int32 => [b'i'.cast_signed(), 0],
        ValueKind::Int64 => [b'q'.cast_signed(), 0],
        ValueKind::Float32 => [b'f'.cast_signed(), 0],
        ValueKind::Float64 => [b'd'.cast_signed(), 0],
    }
}

#[expect(
    clippy::implicit_clone,
    reason = "primitive conversions from `PyO3` borrowed objects need an owned Python object at the FFI boundary"
)]
fn runtime_value_to_object(py: Python<'_>, value: RuntimeValue) -> Py<PyAny> {
    match value {
        RuntimeValue::Bool(value) => {
            let object = value
                .into_pyobject(py)
                .expect("primitive Python conversion should not fail");
            object.to_owned().into_any().unbind()
        }
        RuntimeValue::Int32(value) => {
            let object = value
                .into_pyobject(py)
                .expect("primitive Python conversion should not fail");
            object.to_owned().into_any().unbind()
        }
        RuntimeValue::Int64(value) => {
            let object = value
                .into_pyobject(py)
                .expect("primitive Python conversion should not fail");
            object.to_owned().into_any().unbind()
        }
        RuntimeValue::Float32(value) => {
            let object = value
                .into_pyobject(py)
                .expect("primitive Python conversion should not fail");
            object.to_owned().into_any().unbind()
        }
        RuntimeValue::Float64(value) => {
            let object = value
                .into_pyobject(py)
                .expect("primitive Python conversion should not fail");
            object.to_owned().into_any().unbind()
        }
    }
}

fn py_any_to_runtime_value(
    value: &Bound<'_, PyAny>,
    kind: ValueKind,
    field_name: &str,
) -> PyResult<RuntimeValue> {
    match kind {
        ValueKind::Bool => value
            .extract::<bool>()
            .map(RuntimeValue::Bool)
            .map_err(|_| {
                core_error(CoreError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: kind.as_str(),
                    received: "python object",
                })
            }),
        ValueKind::Int32 => value
            .extract::<i32>()
            .map(RuntimeValue::Int32)
            .map_err(|_| {
                core_error(CoreError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: kind.as_str(),
                    received: "python object",
                })
            }),
        ValueKind::Int64 => value
            .extract::<i64>()
            .map(RuntimeValue::Int64)
            .map_err(|_| {
                core_error(CoreError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: kind.as_str(),
                    received: "python object",
                })
            }),
        ValueKind::Float32 => value
            .extract::<f32>()
            .map(RuntimeValue::Float32)
            .map_err(|_| {
                core_error(CoreError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: kind.as_str(),
                    received: "python object",
                })
            }),
        ValueKind::Float64 => value
            .extract::<f64>()
            .map(RuntimeValue::Float64)
            .map_err(|_| {
                core_error(CoreError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: kind.as_str(),
                    received: "python object",
                })
            }),
    }
}

#[pyclass(module = "arepy_ecs._native")]
pub struct RawWorld {
    inner: SharedWorld,
}

#[pyclass(module = "arepy_ecs._native", unsendable)]
pub struct FieldView {
    inner: SharedWorld,
    component_name: String,
    field_name: String,
    ptr: *mut c_void,
    len: usize,
    byte_len: ffi::Py_ssize_t,
    item_size: ffi::Py_ssize_t,
    shape: [ffi::Py_ssize_t; 1],
    strides: [ffi::Py_ssize_t; 1],
    format: [c_char; 2],
}

impl Drop for FieldView {
    fn drop(&mut self) {
        if let Ok(mut world) = lock_world(&self.inner) {
            let _ = world.unpin_component(&self.component_name);
        }
    }
}

#[pymethods]
impl FieldView {
    fn __len__(&self) -> usize {
        self.len
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldView(component='{}', field='{}', len={})",
            self.component_name, self.field_name, self.len
        )
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "`PyO3` magic methods receive `PyRef` by value to preserve slot signatures"
    )]
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyRuntimeError::new_err("Python requested a null Py_buffer pointer"));
        }

        // SAFETY: CPython passes a valid exporter object for this slot. The raw pointer stored in
        // `FieldView` stays stable because the matching component table is pinned while the view lives.
        if ffi::PyBuffer_FillInfo(view, slf.as_ptr(), slf.ptr, slf.byte_len, 0, flags) != 0 {
            return Err(PyErr::fetch(slf.py()));
        }

        let buffer = &mut *view;
        buffer.itemsize = slf.item_size;
        buffer.ndim = 1;
        if flags & ffi::PyBUF_FORMAT != 0 {
            buffer.format = slf.format.as_ptr().cast_mut();
        }
        if flags & ffi::PyBUF_ND != 0 {
            buffer.shape = slf.shape.as_ptr().cast_mut();
        }
        if flags & ffi::PyBUF_STRIDES != 0 {
            buffer.strides = slf.strides.as_ptr().cast_mut();
        }
        Ok(())
    }

    #[expect(
        clippy::unused_self,
        reason = "`PyO3` buffer protocol methods keep a receiver even when it is not inspected"
    )]
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

#[pymethods]
impl RawWorld {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorldCore::new())),
        }
    }

    fn register_component(&self, name: &str, fields: Vec<(String, String)>) -> PyResult<()> {
        let definitions: Vec<FieldDefinition> = fields
            .into_iter()
            .map(|(field_name, field_kind)| {
                ValueKind::parse(&field_kind).map(|kind| FieldDefinition {
                    name: field_name,
                    kind,
                })
            })
            .collect::<Result<_, _>>()
            .map_err(core_error)?;

        self.inner
            .lock()
            .map_err(|_| lock_error())?
            .register_component(name, definitions)
            .map_err(core_error)
    }

    fn create_entity(&self) -> PyResult<u64> {
        Ok(lock_world(&self.inner)?.create_entity())
    }

    fn kill_entity(&self, entity_id: u64) -> PyResult<()> {
        lock_world(&self.inner)?
            .kill_entity(entity_id)
            .map_err(core_error)
    }

    fn add_component(
        &self,
        entity_id: u64,
        component_name: &str,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let mut world = lock_world(&self.inner)?;
        let schema = world.component_schema(component_name).map_err(core_error)?;
        let mut converted = HashMap::new();

        for field in schema {
            let value = values
                .get_item(field.name.as_str())?
                .ok_or_else(|| {
                    core_error(CoreError::UnknownField {
                        component: component_name.to_string(),
                        field: field.name.clone(),
                    })
                })?;
            converted.insert(
                field.name.clone(),
                py_any_to_runtime_value(&value, field.kind, &field.name)?,
            );
        }

        world
            .add_component(entity_id, component_name, &converted)
            .map_err(core_error)
    }

    fn remove_component(&self, entity_id: u64, component_name: &str) -> PyResult<()> {
        lock_world(&self.inner)?
            .remove_component(entity_id, component_name)
            .map_err(core_error)
    }

    fn has_component(&self, entity_id: u64, component_name: &str) -> PyResult<bool> {
        lock_world(&self.inner)?
            .has_component(entity_id, component_name)
            .map_err(core_error)
    }

    fn get_component(
        &self,
        py: Python<'_>,
        entity_id: u64,
        component_name: &str,
    ) -> PyResult<Py<PyAny>> {
        let component = self
            .inner
            .lock()
            .map_err(|_| lock_error())?
            .component(entity_id, component_name)
            .map_err(core_error)?;
        let dict = PyDict::new(py);
        for (field_name, value) in component {
            dict.set_item(field_name, runtime_value_to_object(py, value))?;
        }
        Ok(dict.into_any().unbind())
    }

    fn get_component_field(
        &self,
        py: Python<'_>,
        entity_id: u64,
        component_name: &str,
        field_name: &str,
    ) -> PyResult<Py<PyAny>> {
        let value = self
            .inner
            .lock()
            .map_err(|_| lock_error())?
            .component_field(entity_id, component_name, field_name)
            .map_err(core_error)?;
        Ok(runtime_value_to_object(py, value))
    }

    fn set_component_field(
        &self,
        entity_id: u64,
        component_name: &str,
        field_name: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut world = lock_world(&self.inner)?;
        let schema = world.component_schema(component_name).map_err(core_error)?;
        let field = schema
            .into_iter()
            .find(|field| field.name == field_name)
            .ok_or_else(|| {
                core_error(CoreError::UnknownField {
                    component: component_name.to_string(),
                    field: field_name.to_string(),
                })
            })?;
        let converted = py_any_to_runtime_value(value, field.kind, field_name)?;
        world
            .set_component_field(entity_id, component_name, field_name, converted)
            .map_err(core_error)
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "`PyO3` extracts Python sequence arguments into owned `Vec<String>` values"
    )]
    fn query_entities(&self, with_components: Vec<String>, without_components: Vec<String>) -> PyResult<Vec<u64>> {
        Ok(lock_world(&self.inner)?.query_entities(&with_components, &without_components))
    }

    fn component_field_view(&self, component_name: &str, field_name: &str) -> PyResult<FieldView> {
        let info = lock_world(&self.inner)?
            .pin_component_field(component_name, field_name)
            .map_err(core_error)?;
        let item_size = py_ssize(info.kind.item_size(), "item_size")?;
        let len = py_ssize(info.len, "len")?;
        let byte_len = py_ssize(info.len.saturating_mul(info.kind.item_size()), "byte_len")?;

        Ok(FieldView {
            inner: Arc::clone(&self.inner),
            component_name: component_name.to_string(),
            field_name: field_name.to_string(),
            ptr: info.ptr,
            len: info.len,
            byte_len,
            item_size,
            shape: [len],
            strides: [item_size],
            format: buffer_format(info.kind),
        })
    }

    fn component_field_values(
        &self,
        py: Python<'_>,
        component_name: &str,
        field_name: &str,
    ) -> PyResult<Py<PyAny>> {
        let snapshot = self
            .inner
            .lock()
            .map_err(|_| lock_error())?
            .component_field_values(component_name, field_name)
            .map_err(core_error)?;

        let list = match snapshot {
            FieldSnapshot::Bool(values) => PyList::new(py, values)?,
            FieldSnapshot::Int32(values) => PyList::new(py, values)?,
            FieldSnapshot::Int64(values) => PyList::new(py, values)?,
            FieldSnapshot::Float32(values) => PyList::new(py, values)?,
            FieldSnapshot::Float64(values) => PyList::new(py, values)?,
        };
        Ok(list.into_any().unbind())
    }

    fn alive_count(&self) -> PyResult<usize> {
        Ok(lock_world(&self.inner)?.alive_count())
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RawWorld>()?;
    module.add_class::<FieldView>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}