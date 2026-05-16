//! Dense Rust storage for the Python-facing ECS API.
//!
//! This crate owns entity allocation, component column storage, query filtering,
//! and pinned field exports used by the `PyO3` bridge.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ffi::c_void;

use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

/// Errors emitted by the ECS storage layer.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    #[error("component schema not registered: {0}")]
    MissingSchema(String),
    #[error("entity `{0}` not found")]
    MissingEntity(u64),
    #[error("component `{component}` is not attached to entity `{entity}`")]
    MissingComponent { entity: u64, component: String },
    #[error("field `{field}` not found in component `{component}`")]
    UnknownField { component: String, field: String },
    #[error("component `{0}` already exists with a different schema")]
    SchemaConflict(String),
    #[error("unsupported field kind `{0}`")]
    UnsupportedFieldKind(String),
    #[error("field `{field}` expected {expected} but received {received}")]
    FieldTypeMismatch {
        field: String,
        expected: &'static str,
        received: &'static str,
    },
    #[error("component `{0}` has active exported views; release them before changing its layout")]
    ActiveViews(String),
    #[error(
        "storage invariant violated for component `{component}` and entity `{entity}` at row `{row}`"
    )]
    StorageInvariantViolation {
        component: String,
        entity: u64,
        row: usize,
    },
}

/// Scalar kinds supported by the current dense column storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueKind {
    Bool,
    Int32,
    Float32,
}

impl ValueKind {
    /// Parses a Python-side field annotation into a supported storage kind.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::UnsupportedFieldKind`] when the annotation string is
    /// not mapped to one of the scalar storage kinds supported by the core.
    pub fn parse(name: &str) -> CoreResult<Self> {
        match name {
            "bool" | "Bool" | "numpy.bool_" | "np.bool_" => Ok(Self::Bool),
            "int32" | "Int32" | "int" | "numpy.int32" | "np.int32" => Ok(Self::Int32),
            "float32" | "Float32" | "float" | "numpy.float32" | "np.float32" => Ok(Self::Float32),
            other => Err(CoreError::UnsupportedFieldKind(other.to_string())),
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int32 => "int32",
            Self::Float32 => "float32",
        }
    }

    #[must_use]
    pub fn item_size(self) -> usize {
        match self {
            Self::Bool => std::mem::size_of::<bool>(),
            Self::Int32 => std::mem::size_of::<i32>(),
            Self::Float32 => std::mem::size_of::<f32>(),
        }
    }
}

/// Runtime payload used to move scalar values between Python bindings and Rust storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RuntimeValue {
    Bool(bool),
    Int32(i32),
    Float32(f32),
}

impl RuntimeValue {
    #[must_use]
    pub fn kind(self) -> ValueKind {
        match self {
            Self::Bool(_) => ValueKind::Bool,
            Self::Int32(_) => ValueKind::Int32,
            Self::Float32(_) => ValueKind::Float32,
        }
    }

    #[must_use]
    pub fn type_name(self) -> &'static str {
        self.kind().as_str()
    }
}

/// Declares one named field within a component schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDefinition {
    pub name: String,
    pub kind: ValueKind,
}

/// Raw pointer metadata for a pinned exported field buffer.
#[derive(Clone, Copy, Debug)]
pub struct FieldBufferInfo {
    pub ptr: *mut c_void,
    pub len: usize,
    pub kind: ValueKind,
}

/// Snapshot copy of a whole component field column.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldSnapshot {
    Bool(Vec<bool>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
}

#[derive(Debug)]
enum FieldColumn {
    Bool(Vec<bool>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
}

impl FieldColumn {
    fn new(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Bool => Self::Bool(Vec::new()),
            ValueKind::Int32 => Self::Int32(Vec::new()),
            ValueKind::Float32 => Self::Float32(Vec::new()),
        }
    }

    fn push(&mut self, value: &RuntimeValue, field_name: &str) -> CoreResult<()> {
        match (self, value) {
            (Self::Bool(values), RuntimeValue::Bool(value)) => values.push(*value),
            (Self::Int32(values), RuntimeValue::Int32(value)) => values.push(*value),
            (Self::Float32(values), RuntimeValue::Float32(value)) => values.push(*value),
            (column, received) => {
                return Err(CoreError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: column.kind().as_str(),
                    received: received.type_name(),
                });
            }
        }
        Ok(())
    }

    fn get(&self, row: usize) -> RuntimeValue {
        match self {
            Self::Bool(values) => RuntimeValue::Bool(values[row]),
            Self::Int32(values) => RuntimeValue::Int32(values[row]),
            Self::Float32(values) => RuntimeValue::Float32(values[row]),
        }
    }

    fn set(&mut self, row: usize, value: &RuntimeValue, field_name: &str) -> CoreResult<()> {
        match (self, value) {
            (Self::Bool(values), RuntimeValue::Bool(value)) => values[row] = *value,
            (Self::Int32(values), RuntimeValue::Int32(value)) => values[row] = *value,
            (Self::Float32(values), RuntimeValue::Float32(value)) => values[row] = *value,
            (column, received) => {
                return Err(CoreError::FieldTypeMismatch {
                    field: field_name.to_string(),
                    expected: column.kind().as_str(),
                    received: received.type_name(),
                });
            }
        }
        Ok(())
    }

    fn swap_remove(&mut self, row: usize) {
        match self {
            Self::Bool(values) => {
                values.swap_remove(row);
            }
            Self::Int32(values) => {
                values.swap_remove(row);
            }
            Self::Float32(values) => {
                values.swap_remove(row);
            }
        }
    }

    fn kind(&self) -> ValueKind {
        match self {
            Self::Bool(_) => ValueKind::Bool,
            Self::Int32(_) => ValueKind::Int32,
            Self::Float32(_) => ValueKind::Float32,
        }
    }

    fn snapshot(&self) -> FieldSnapshot {
        match self {
            Self::Bool(values) => FieldSnapshot::Bool(values.clone()),
            Self::Int32(values) => FieldSnapshot::Int32(values.clone()),
            Self::Float32(values) => FieldSnapshot::Float32(values.clone()),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Float32(values) => values.len(),
        }
    }

    fn data_ptr(&mut self) -> *mut c_void {
        match self {
            Self::Bool(values) => values.as_mut_ptr().cast(),
            Self::Int32(values) => values.as_mut_ptr().cast(),
            Self::Float32(values) => values.as_mut_ptr().cast(),
        }
    }
}

#[derive(Debug)]
struct ComponentTable {
    name: String,
    fields: Vec<FieldDefinition>,
    entity_ids: Vec<u64>,
    rows: HashMap<u64, usize>,
    columns: Vec<FieldColumn>,
    active_view_count: usize,
}

impl ComponentTable {
    fn new(name: &str, fields: Vec<FieldDefinition>) -> Self {
        let columns = fields.iter().map(|field| FieldColumn::new(field.kind)).collect();
        Self {
            name: name.to_string(),
            fields,
            entity_ids: Vec::new(),
            rows: HashMap::new(),
            columns,
            active_view_count: 0,
        }
    }

    fn same_schema(&self, other: &[FieldDefinition]) -> bool {
        self.fields == other
    }

    fn field_index(&self, field_name: &str) -> CoreResult<usize> {
        self.fields
            .iter()
            .position(|field| field.name == field_name)
            .ok_or_else(|| CoreError::UnknownField {
                component: self.name.clone(),
                field: field_name.to_string(),
            })
    }

    fn row_for_entity(&self, entity_id: u64) -> CoreResult<usize> {
        self.rows
            .get(&entity_id)
            .copied()
            .ok_or_else(|| CoreError::MissingComponent {
                entity: entity_id,
                component: self.name.clone(),
            })
    }

    fn ensure_row_in_bounds(&self, entity_id: u64, row: usize) -> CoreResult<()> {
        let is_consistent = row < self.entity_ids.len() && self.columns.iter().all(|column| row < column.len());
        if is_consistent {
            Ok(())
        } else {
            Err(CoreError::StorageInvariantViolation {
                component: self.name.clone(),
                entity: entity_id,
                row,
            })
        }
    }

    fn insert_or_replace(
        &mut self,
        entity_id: u64,
        values: &HashMap<String, RuntimeValue>,
    ) -> CoreResult<()> {
        for key in values.keys() {
            if self.fields.iter().all(|field| field.name != *key) {
                return Err(CoreError::UnknownField {
                    component: self.name.clone(),
                    field: key.clone(),
                });
            }
        }

        if let Some(&row) = self.rows.get(&entity_id) {
            for (index, field) in self.fields.iter().enumerate() {
                let value = values.get(&field.name).ok_or_else(|| CoreError::UnknownField {
                    component: self.name.clone(),
                    field: field.name.clone(),
                })?;
                self.columns[index].set(row, value, &field.name)?;
            }
            return Ok(());
        }

        self.ensure_no_active_views()?;

        for (index, field) in self.fields.iter().enumerate() {
            let value = values.get(&field.name).ok_or_else(|| CoreError::UnknownField {
                component: self.name.clone(),
                field: field.name.clone(),
            })?;
            self.columns[index].push(value, &field.name)?;
        }

        self.rows.insert(entity_id, self.entity_ids.len());
        self.entity_ids.push(entity_id);
        Ok(())
    }

    fn remove(&mut self, entity_id: u64) -> CoreResult<()> {
        self.ensure_no_active_views()?;
        let row = self
            .rows
            .remove(&entity_id)
            .ok_or_else(|| CoreError::MissingComponent {
                entity: entity_id,
                component: self.name.clone(),
            })?;

        for column in &mut self.columns {
            column.swap_remove(row);
        }
        self.entity_ids.swap_remove(row);

        if row < self.entity_ids.len() {
            let moved_entity = self.entity_ids[row];
            self.rows.insert(moved_entity, row);
        }
        Ok(())
    }

    fn contains(&self, entity_id: u64) -> bool {
        self.rows.contains_key(&entity_id)
    }

    fn component(&self, entity_id: u64) -> CoreResult<BTreeMap<String, RuntimeValue>> {
        let row = self.row_for_entity(entity_id)?;
        self.ensure_row_in_bounds(entity_id, row)?;

        let mut values = BTreeMap::new();
        for (index, field) in self.fields.iter().enumerate() {
            values.insert(field.name.clone(), self.columns[index].get(row));
        }
        Ok(values)
    }

    fn field_value(&self, entity_id: u64, field_name: &str) -> CoreResult<RuntimeValue> {
        let row = self.row_for_entity(entity_id)?;
        self.ensure_row_in_bounds(entity_id, row)?;
        let index = self.field_index(field_name)?;
        Ok(self.columns[index].get(row))
    }

    fn set_field_value(
        &mut self,
        entity_id: u64,
        field_name: &str,
        value: RuntimeValue,
    ) -> CoreResult<()> {
        let row = self.row_for_entity(entity_id)?;
        self.ensure_row_in_bounds(entity_id, row)?;
        let index = self.field_index(field_name)?;

        self.columns[index].set(row, &value, field_name)
    }

    fn field_snapshot(&self, field_name: &str) -> CoreResult<FieldSnapshot> {
        let index = self.field_index(field_name)?;
        Ok(self.columns[index].snapshot())
    }

    fn pin_field(&mut self, field_name: &str) -> CoreResult<FieldBufferInfo> {
        let index = self.field_index(field_name)?;
        self.active_view_count += 1;
        let column = &mut self.columns[index];
        Ok(FieldBufferInfo {
            ptr: column.data_ptr(),
            len: column.len(),
            kind: column.kind(),
        })
    }

    fn unpin(&mut self) {
        self.active_view_count = self.active_view_count.saturating_sub(1);
    }

    fn ensure_no_active_views(&self) -> CoreResult<()> {
        if self.active_view_count == 0 {
            Ok(())
        } else {
            Err(CoreError::ActiveViews(self.name.clone()))
        }
    }
}

/// Owns entities, component tables, and query matching state for the ECS core.
#[derive(Debug, Default)]
pub struct WorldCore {
    next_entity_id: u64,
    free_entity_ids: VecDeque<u64>,
    alive_entities: HashSet<u64>,
    entity_components: HashMap<u64, HashSet<String>>,
    component_tables: HashMap<String, ComponentTable>,
}

impl WorldCore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn table(&self, component_name: &str) -> CoreResult<&ComponentTable> {
        self.component_tables
            .get(component_name)
            .ok_or_else(|| CoreError::MissingSchema(component_name.to_string()))
    }

    fn table_mut(&mut self, component_name: &str) -> CoreResult<&mut ComponentTable> {
        self.component_tables
            .get_mut(component_name)
            .ok_or_else(|| CoreError::MissingSchema(component_name.to_string()))
    }

    /// Registers a component schema if it has not been seen before.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::SchemaConflict`] when the component name already
    /// exists with a different schema.
    pub fn register_component(
        &mut self,
        name: &str,
        fields: Vec<FieldDefinition>,
    ) -> CoreResult<()> {
        if let Some(existing) = self.component_tables.get(name) {
            if existing.same_schema(&fields) {
                return Ok(());
            }
            return Err(CoreError::SchemaConflict(name.to_string()));
        }

        self.component_tables
            .insert(name.to_string(), ComponentTable::new(name, fields));
        Ok(())
    }

    /// Returns the field definitions for a registered component.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::MissingSchema`] when the component has not been registered.
    pub fn component_schema(&self, component_name: &str) -> CoreResult<Vec<FieldDefinition>> {
        Ok(self.table(component_name)?.fields.clone())
    }

    /// Allocates a new entity identifier, reusing freed identifiers when possible.
    pub fn create_entity(&mut self) -> u64 {
        let entity_id = self.free_entity_ids.pop_front().unwrap_or_else(|| {
            let current = self.next_entity_id;
            self.next_entity_id += 1;
            current
        });
        self.alive_entities.insert(entity_id);
        self.entity_components.entry(entity_id).or_default();
        entity_id
    }

    /// Removes an entity and all of its components.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::MissingEntity`] when the entity is dead or unknown.
    /// Returns [`CoreError::ActiveViews`] when one of the entity's component tables
    /// is pinned by an exported Python buffer.
    pub fn kill_entity(&mut self, entity_id: u64) -> CoreResult<()> {
        self.ensure_entity(entity_id)?;
        let component_names = self
            .entity_components
            .get(&entity_id)
            .cloned()
            .unwrap_or_default();
        for component_name in &component_names {
            if let Some(table) = self.component_tables.get(component_name) {
                table.ensure_no_active_views()?;
            }
        }

        let component_names = self.entity_components.remove(&entity_id).unwrap_or_default();
        for component_name in component_names {
            if let Some(table) = self.component_tables.get_mut(&component_name) {
                let _ = table.remove(entity_id);
            }
        }
        self.alive_entities.remove(&entity_id);
        self.free_entity_ids.push_back(entity_id);
        Ok(())
    }

    /// Inserts or replaces one component instance for an entity.
    ///
    /// # Errors
    ///
    /// Returns an error when the entity or schema is missing, when the payload does
    /// not match the declared schema, or when the table is pinned by an active view.
    pub fn add_component(
        &mut self,
        entity_id: u64,
        component_name: &str,
        values: &HashMap<String, RuntimeValue>,
    ) -> CoreResult<()> {
        self.ensure_entity(entity_id)?;
        let table = self.table_mut(component_name)?;
        table.insert_or_replace(entity_id, values)?;
        self.entity_components
            .entry(entity_id)
            .or_default()
            .insert(component_name.to_string());
        Ok(())
    }

    /// Detaches a component from an entity.
    ///
    /// # Errors
    ///
    /// Returns an error when the entity, schema, or concrete component row is missing,
    /// or when the table is pinned by an active exported view.
    pub fn remove_component(&mut self, entity_id: u64, component_name: &str) -> CoreResult<()> {
        self.ensure_entity(entity_id)?;
        let table = self.table_mut(component_name)?;
        table.remove(entity_id)?;
        if let Some(components) = self.entity_components.get_mut(&entity_id) {
            components.remove(component_name);
        }
        Ok(())
    }

    /// Checks whether an entity currently owns a component of the requested type.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::MissingEntity`] when the entity is dead or unknown.
    pub fn has_component(&self, entity_id: u64, component_name: &str) -> CoreResult<bool> {
        self.ensure_entity(entity_id)?;
        Ok(self
            .component_tables
            .get(component_name)
            .is_some_and(|table| table.contains(entity_id)))
    }

    /// Returns a field-by-field snapshot of one component attached to an entity.
    ///
    /// # Errors
    ///
    /// Returns an error when the entity, schema, or component row is missing.
    pub fn component(
        &self,
        entity_id: u64,
        component_name: &str,
    ) -> CoreResult<BTreeMap<String, RuntimeValue>> {
        self.ensure_entity(entity_id)?;
        self.table(component_name)?.component(entity_id)
    }

    /// Reads one scalar field from a component attached to an entity.
    ///
    /// # Errors
    ///
    /// Returns an error when the entity, schema, component row, or field is missing.
    pub fn component_field(
        &self,
        entity_id: u64,
        component_name: &str,
        field_name: &str,
    ) -> CoreResult<RuntimeValue> {
        self.ensure_entity(entity_id)?;
        self.table(component_name)?.field_value(entity_id, field_name)
    }

    /// Writes one scalar field inside a component row.
    ///
    /// # Errors
    ///
    /// Returns an error when the entity, schema, component row, or field is missing,
    /// or when the value kind does not match the declared field type.
    pub fn set_component_field(
        &mut self,
        entity_id: u64,
        component_name: &str,
        field_name: &str,
        value: RuntimeValue,
    ) -> CoreResult<()> {
        self.ensure_entity(entity_id)?;
        self.table_mut(component_name)?
            .set_field_value(entity_id, field_name, value)
    }

    /// Returns all entity identifiers matching the requested component filters.
    #[must_use]
    pub fn query_entities(&self, with_components: &[String], without_components: &[String]) -> Vec<u64> {
        let mut entity_ids: Vec<u64> = self
            .alive_entities
            .iter()
            .copied()
            .filter(|entity_id| {
                let components = self.entity_components.get(entity_id);
                with_components
                    .iter()
                    .all(|component| components.is_some_and(|set| set.contains(component)))
                    && without_components
                        .iter()
                        .all(|component| !components.is_some_and(|set| set.contains(component)))
            })
            .collect();
        entity_ids.sort_unstable();
        entity_ids
    }

    /// Copies an entire scalar field column into a snapshot enum.
    ///
    /// # Errors
    ///
    /// Returns an error when the component schema or field is missing.
    pub fn component_field_values(
        &self,
        component_name: &str,
        field_name: &str,
    ) -> CoreResult<FieldSnapshot> {
        self.table(component_name)?.field_snapshot(field_name)
    }

    /// Pins a component field column and returns raw buffer metadata for Python.
    ///
    /// # Errors
    ///
    /// Returns an error when the component schema or field is missing.
    pub fn pin_component_field(
        &mut self,
        component_name: &str,
        field_name: &str,
    ) -> CoreResult<FieldBufferInfo> {
        self.table_mut(component_name)?.pin_field(field_name)
    }

    /// Releases one active pin for a component table previously exported to Python.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::MissingSchema`] when the component is unknown.
    pub fn unpin_component(&mut self, component_name: &str) -> CoreResult<()> {
        let table = self.table_mut(component_name)?;
        table.unpin();
        Ok(())
    }

    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.alive_entities.len()
    }

    fn ensure_entity(&self, entity_id: u64) -> CoreResult<()> {
        if self.alive_entities.contains(&entity_id) {
            Ok(())
        } else {
            Err(CoreError::MissingEntity(entity_id))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CoreError, FieldDefinition, RuntimeValue, ValueKind, WorldCore};
    use std::collections::HashMap;

    #[test]
    fn query_entities_returns_only_matching_entities() {
        let mut world = WorldCore::new();
        world
            .register_component(
                "Position",
                vec![
                    FieldDefinition {
                        name: "x".to_string(),
                        kind: ValueKind::Float32,
                    },
                    FieldDefinition {
                        name: "y".to_string(),
                        kind: ValueKind::Float32,
                    },
                ],
            )
            .unwrap();
        world
            .register_component(
                "Velocity",
                vec![FieldDefinition {
                    name: "x".to_string(),
                    kind: ValueKind::Float32,
                }],
            )
            .unwrap();

        let moving = world.create_entity();
        let static_entity = world.create_entity();

        world
            .add_component(
                moving,
                "Position",
                &HashMap::from([
                    ("x".to_string(), RuntimeValue::Float32(1.0)),
                    ("y".to_string(), RuntimeValue::Float32(2.0)),
                ]),
            )
            .unwrap();
        world
            .add_component(
                moving,
                "Velocity",
                &HashMap::from([("x".to_string(), RuntimeValue::Float32(0.5))]),
            )
            .unwrap();
        world
            .add_component(
                static_entity,
                "Position",
                &HashMap::from([
                    ("x".to_string(), RuntimeValue::Float32(4.0)),
                    ("y".to_string(), RuntimeValue::Float32(5.0)),
                ]),
            )
            .unwrap();

        let entity_ids = world.query_entities(
            &["Position".to_string()],
            &["Velocity".to_string()],
        );

        assert_eq!(entity_ids, vec![static_entity]);
    }

    #[test]
    fn add_component_returns_active_view_error_when_table_is_pinned() {
        let mut world = WorldCore::new();
        world
            .register_component(
                "Position",
                vec![FieldDefinition {
                    name: "x".to_string(),
                    kind: ValueKind::Float32,
                }],
            )
            .unwrap();

        let entity = world.create_entity();
        world
            .add_component(
                entity,
                "Position",
                &HashMap::from([("x".to_string(), RuntimeValue::Float32(1.0))]),
            )
            .unwrap();

        let view = world.pin_component_field("Position", "x").unwrap();
        assert_eq!(view.len, 1);

        let other = world.create_entity();
        let error = world
            .add_component(
                other,
                "Position",
                &HashMap::from([("x".to_string(), RuntimeValue::Float32(2.0))]),
            )
            .unwrap_err();
        assert_eq!(error, CoreError::ActiveViews("Position".to_string()));

        world.unpin_component("Position").unwrap();
        world
            .add_component(
                other,
                "Position",
                &HashMap::from([("x".to_string(), RuntimeValue::Float32(2.0))]),
            )
            .unwrap();
    }
}