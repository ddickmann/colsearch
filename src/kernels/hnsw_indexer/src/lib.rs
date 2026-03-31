use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyList}; 
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray, PyArray1, PyArrayMethods};
use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use segment::entry::SegmentEntry; 
use segment::vector_storage::VectorStorage;
use segment::data_types::vectors::QueryVector;
use segment::data_types::query_context::QueryContext;
use common::counter::hardware_accumulator::HwMeasurementAcc;

use std::str::FromStr;
use segment::types::{
    Distance, PointIdType, PayloadKeyType,
};
use segment::segment::Segment;
use segment::segment_constructor::{build_segment, load_segment};
use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
use segment::index::PayloadIndex; 

/// Python wrapper for Qdrant's Segment
#[pyclass]
pub struct HnswSegment {
    segment: Segment,
    dim: usize,
    path: PathBuf,
}

#[pymethods]
impl HnswSegment {
    #[new]
    #[pyo3(signature = (path, dim, distance_metric = "cosine", m = 16, ef_construct = 100, on_disk = false, is_appendable = true, multivector_comparator = None))]
    fn new(
        path: String,
        dim: usize,
        distance_metric: &str,
        m: usize,
        ef_construct: usize,
        on_disk: bool,
        is_appendable: bool,
        multivector_comparator: Option<&str>,
    ) -> PyResult<Self> {
        let path_buf = PathBuf::from(&path);
        if !path_buf.exists() {
            std::fs::create_dir_all(&path_buf)
                .map_err(|e| PyValueError::new_err(format!("Failed to create directory: {}", e)))?;
        }

        let storage_type = if on_disk {
            segment::types::VectorStorageType::ChunkedMmap
        } else {
            segment::types::VectorStorageType::InRamChunkedMmap
        };
        
        let mut index_config = if is_appendable {
            segment::types::Indexes::Plain {}
        } else {
            let hnsw_config = segment::types::HnswConfig {
                m,
                ef_construct,
                full_scan_threshold: 10000,
                max_indexing_threads: 0,
                on_disk: Some(on_disk),
                payload_m: None,
                inline_storage: None,
            };
            segment::types::Indexes::Hnsw(hnsw_config)
        };

        // MultiVector Configuration
        let multivector_config = if let Some(comp) = multivector_comparator {
             let comparator = match comp {
                 "max_sim" => segment::types::MultiVectorComparator::MaxSim,
                 _ => return Err(PyValueError::new_err("Invalid multivector comparator")),
             };
             Some(segment::types::VectorDataConfig {
                 size: dim,
                 distance: match distance_metric {
                     "cosine" => segment::types::Distance::Cosine,
                     "dot" => segment::types::Distance::Dot,
                     "euclidean" => segment::types::Distance::Euclid,
                     _ => return Err(PyValueError::new_err("Invalid distance metric")),
                 },
                 storage_type: storage_type.clone(),
                 index: index_config.clone(),
                 quantization_config: None,
                 multivector_config: Some(segment::types::MultiVectorConfig {
                     comparator,
                 }),
                 datatype: None,
             })
        } else {
             None
        };
        
        // Final Config Logic
        // If multivector_config is set, use it. Otherwise use standard config.
        let vector_data_config = if let Some(cfg) = multivector_config {
            cfg
        } else {
            segment::types::VectorDataConfig {
                size: dim,
                distance: match distance_metric {
                    "cosine" => segment::types::Distance::Cosine,
                    "dot" => segment::types::Distance::Dot,
                    "euclidean" => segment::types::Distance::Euclid,
                    _ => return Err(PyValueError::new_err("Invalid distance metric")),
                },
                storage_type,
                index: index_config,
                quantization_config: None,
                multivector_config: None,
                datatype: None,
            }
        };

        let config = segment::types::SegmentConfig {
            vector_data: HashMap::from([(
                segment::data_types::vectors::DEFAULT_VECTOR_NAME.to_owned(),
                vector_data_config
            )]),
            sparse_vector_data: Default::default(),
            payload_storage_type: if on_disk || !is_appendable {
                segment::types::PayloadStorageType::OnDisk
            } else {
                segment::types::PayloadStorageType::InMemory
            },
        };

        let existing_segment_path = if path_buf.join("segment.json").exists() {
            Some(path_buf.clone())
        } else {
            std::fs::read_dir(&path_buf)
                .ok()
                .and_then(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .find(|candidate| candidate.is_dir() && candidate.join("segment.json").exists())
                })
        };

        let segment = if let Some(existing_path) = existing_segment_path {
            load_segment(&existing_path, &AtomicBool::new(false))
                .map_err(|e| PyValueError::new_err(format!("Failed to load segment: {}", e)))?
                .ok_or_else(|| PyValueError::new_err("Existing segment is not ready or was marked deleted"))?
        } else {
            build_segment(&path_buf, &config, true)
                .map_err(|e| PyValueError::new_err(format!("Failed to build segment: {}", e)))?
        };

        Ok(Self {
            segment,
            dim,
            path: path_buf,
        })
    }

    /// Add vectors with optional payloads
    #[pyo3(signature = (vectors, ids = None, payloads = None))]
    fn add(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: Option<Vec<u64>>,
        payloads: Option<Vec<Bound<'_, PyDict>>>,
    ) -> PyResult<Vec<u64>> {
        let vectors_array = vectors.as_array();
        let (n, d) = (vectors_array.shape()[0], vectors_array.shape()[1]);

        if d != self.dim {
            return Err(PyValueError::new_err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim, d
            )));
        }

        let mut point_ids = Vec::new();
        let segment_mut = &mut self.segment;
        let hw_counter = HwMeasurementAcc::disposable(); 
        
        let first_op_num = segment_mut.version() + 1;

        for i in 0..n {
            let point_id = ids.as_ref().map(|v| v[i])
                .unwrap_or_else(|| rand::random());
            let op_num = first_op_num + i as u64;
            
            let vector_slice = vectors_array.row(i);
            let named_vectors = segment::data_types::named_vectors::NamedVectors::from_ref(
                DEFAULT_VECTOR_NAME,
                vector_slice.as_slice().unwrap().into(),
            );
            
            let payload = payloads.as_ref()
                .and_then(|p| p.get(i))
                .map(|dict| dict_to_payload(dict))
                .transpose()?;
                
            segment_mut.upsert_point(
                op_num,
                PointIdType::NumId(point_id),
                named_vectors,
                &hw_counter.get_counter_cell(),
            ).map_err(|e| PyValueError::new_err(format!("Failed to insert point: {}", e)))?;

            if let Some(payload) = payload {
                let payload_struct: segment::types::Payload = payload.into();
                segment_mut.set_full_payload(
                    op_num,
                    PointIdType::NumId(point_id),
                    &payload_struct,
                    &hw_counter.get_counter_cell(),
                ).map_err(|e| PyValueError::new_err(format!("Failed to set payload: {}", e)))?;
            }

            point_ids.push(point_id);
        }

        Ok(point_ids)
    }

    /// Add MultiDense vectors
    /// 
    /// inputs:
    ///   vectors: List of 2D numpy arrays. Each array is (M_i, D) where M_i is number of vectors for point i.
    #[pyo3(signature = (vectors, ids = None, payloads = None))]
    fn add_multidense(
        &mut self,
        vectors: Vec<PyReadonlyArray2<f32>>, 
        ids: Option<Vec<u64>>,
        payloads: Option<Vec<Bound<'_, PyDict>>>,
    ) -> PyResult<Vec<u64>> {
        let n = vectors.len();
        
        let mut point_ids = Vec::new();
        let segment_mut = &mut self.segment;
        let hw_counter = HwMeasurementAcc::disposable();
        let first_op_num = segment_mut.version() + 1;

        for i in 0..n {
             let vec_array = vectors[i].as_array();
             let (m, d) = (vec_array.shape()[0], vec_array.shape()[1]);
             
             if d != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Vector dimension mismatch at index {}: expected {}, got {}",
                    i, self.dim, d
                )));
             }
             
             // Create MultiDenseVector
             // Needs flattened slice
             // Warning: to_owned() copies, but we need contiguous memory for MultiDenseVector if stride is not C
             // as_slice() on standard numpy array should be 0-copy if C-contiguous.
             let flattened = vec_array.as_slice().ok_or_else(|| PyValueError::new_err("Vector array not C-contiguous"))?;
             
             let multi_dense = segment::data_types::vectors::MultiDenseVectorInternal::new(flattened.to_vec(), d);
             let cow_multi = segment::data_types::named_vectors::CowMultiVector::Owned(multi_dense);
             let cow_vector = segment::data_types::named_vectors::CowVector::MultiDense(cow_multi);
             
             let named_vectors = segment::data_types::named_vectors::NamedVectors::from_ref(
                 DEFAULT_VECTOR_NAME,
                 cow_vector.as_vec_ref(),
             );
             
             let point_id = ids.as_ref().map(|v| v[i])
                .unwrap_or_else(|| rand::random());
             let op_num = first_op_num + i as u64;

             let payload = payloads.as_ref()
                .and_then(|p| p.get(i))
                .map(|dict| dict_to_payload(dict))
                .transpose()?;

             segment_mut.upsert_point(
                op_num,
                PointIdType::NumId(point_id),
                named_vectors,
                &hw_counter.get_counter_cell(),
            ).map_err(|e| PyValueError::new_err(format!("Failed to insert point: {}", e)))?;

            if let Some(payload) = payload {
                let payload_struct: segment::types::Payload = payload.into();
                segment_mut.set_full_payload(
                    op_num,
                    PointIdType::NumId(point_id),
                    &payload_struct,
                    &hw_counter.get_counter_cell(),
                ).map_err(|e| PyValueError::new_err(format!("Failed to set payload: {}", e)))?;
            }

            point_ids.push(point_id);
        }
        
        Ok(point_ids)
    }

    /// Delete points by ID
    #[pyo3(signature = (ids))]
    fn delete(&mut self, ids: Vec<u64>) -> PyResult<usize> {
        let segment_mut = &mut self.segment;
        let hw_counter = HwMeasurementAcc::disposable();
        let op_num = segment_mut.version() + 1;
        let mut deleted_count = 0;

        for id in ids {
            let point_id = PointIdType::NumId(id);
            let res = segment_mut.delete_point(
                op_num,
                point_id,
                &hw_counter.get_counter_cell(),
            );
            
            if let Ok(true) = res {
                deleted_count += 1;
            }
        }
        
        Ok(deleted_count)
    }
    
    /// Upsert payload for a specific point
    #[pyo3(signature = (id, payload))]
    fn upsert_payload(&mut self, id: u64, payload: Bound<'_, PyDict>) -> PyResult<()> {
        let segment_mut = &mut self.segment;
        let hw_counter = HwMeasurementAcc::disposable();
        let op_num = segment_mut.version() + 1;
        let point_id = PointIdType::NumId(id);
        
        let payload_map = dict_to_payload(&payload)?;
        let payload_struct: segment::types::Payload = payload_map.into();
        
        segment_mut.set_payload(
            op_num,
            point_id,
            &payload_struct,
            &None,
            &hw_counter.get_counter_cell(),
        ).map_err(|e| PyValueError::new_err(format!("Failed to upsert payload: {}", e)))?;
        
        Ok(())
    }

    /// Retrieve vector and payload by ID
    /// 
    /// Used by Knapsack Solver to fetch candidate details.
    /// Returns: List of (id, vector_numpy, payload_dict) tuples
    #[pyo3(signature = (ids))]
    fn retrieve(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<Vec<(u64, Option<PyObject>, Option<PyObject>)>> {
        let id_tracker = self.segment.id_tracker.borrow();
        let payload_index = self.segment.payload_index.borrow();
        let vector_storage = self.segment.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
        let hw_counter = HwMeasurementAcc::disposable();
        
        let mut results = Vec::with_capacity(ids.len());
        
        for id in ids {
            let point_id = PointIdType::NumId(id);
            let internal_id = id_tracker.internal_id(point_id);
            
            if let Some(iid) = internal_id {
                // Fetch Vector
                // Use get_vector which returns CowVector
                let vector_cow = vector_storage.get_vector_opt::<segment::vector_storage::Random>(iid);
                let vector_py = if let Some(cow) = vector_cow {
                     match cow {
                        segment::data_types::named_vectors::CowVector::Dense(c) => {
                             let vec_slice: &[f32] = c.as_ref(); 
                             let numpy_arr = vec_slice.to_pyarray_bound(py);
                             Some(numpy_arr.into())
                        },
                        segment::data_types::named_vectors::CowVector::MultiDense(c) => {
                             let multi_dense = c.as_vec_ref();
                             let flattened: &[f32] = multi_dense.flattened_vectors;
                             let dim = multi_dense.dim;
                             let count = flattened.len() / dim;
                             let flat_arr = PyArray1::from_slice_bound(py, flattened);
                             let numpy_arr = flat_arr.reshape((count, dim)).unwrap();
                             Some(numpy_arr.into())
                        },
                         segment::data_types::named_vectors::CowVector::Sparse(c) => {
                             let sparse = c.as_ref();
                             let indices = sparse.indices.to_pyarray_bound(py);
                             let values = sparse.values.to_pyarray_bound(py);
                             let tuple = (indices, values).to_object(py);
                             Some(tuple)
                        },
                        // _ => None 
                     }
                } else {
                    None
                };

                // Fetch Payload
                let payload_res = payload_index.get_payload(iid, &hw_counter.get_counter_cell());
                let payload_py = if let Ok(payload_struct) = payload_res {
                     let map = payload_struct.0;
                     Some(json_to_python_dict(py, &map)?)
                } else {
                     None
                };
                
                results.push((id, vector_py, payload_py));
            } else {
                // Point not found
                results.push((id, None, None));
            }
        }
        
        Ok(results)
    }

    /// Search for nearest neighbors with optional payload filtering
    #[pyo3(signature = (query, k = 10, ef = None, filter = None))]
    fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef: Option<usize>,
        filter: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let query_vec = query.as_slice()?;
        
        if query_vec.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dim,
                query_vec.len()
            )));
        }

        let query_vector = QueryVector::from(query_vec.to_vec());
        
        let filter_condition = filter
            .map(|f| dict_to_filter(&f))
            .transpose()?;

        let search_params = segment::types::SearchParams {
            hnsw_ef: ef,
            exact: false, 
            quantization: None,
            indexed_only: false,
            acorn: None,
        };

        let hw_acc = HwMeasurementAcc::disposable();
        let query_context = QueryContext::new(10000, hw_acc);
        let segment_query_context = query_context.get_segment_query_context();

        let results = self.segment
            .search_batch(
                DEFAULT_VECTOR_NAME,
                &[&query_vector],
                &segment::types::WithPayload::default(),
                &segment::types::WithVector::Bool(false),
                filter_condition.as_ref(),
                k,
                Some(&search_params),
                &segment_query_context,
            )
            .map_err(|e| PyValueError::new_err(format!("Search failed: {}", e)))?;

        let batch_results = results.into_iter().next().unwrap_or_default();

        Ok(batch_results
            .into_iter()
            .map(|sp| {
                let id = match sp.id {
                    PointIdType::NumId(n) => n,
                    PointIdType::Uuid(_) => 0, 
                };
                (id, sp.score)
            })
            .collect())
    }

    /// GEM-style multi-entry search using cluster representative entry points.
    ///
    /// Uses per-entry candidate queues with round-robin expansion and optional
    /// search_set cluster filtering, matching GEM's searchBaseLayerClusterEntriesMulti.
    ///
    /// `entry_point_ids`: external u64 IDs of cluster representatives.
    /// `search_set_ids`: external u64 IDs of all points in the probed clusters.
    ///   If empty, no cluster filter is applied.
    #[pyo3(signature = (query, entry_point_ids, search_set_ids = None, k = 10, ef = None, filter = None))]
    fn search_gem(
        &self,
        query: PyReadonlyArray1<f32>,
        entry_point_ids: Vec<u64>,
        search_set_ids: Option<Vec<u64>>,
        k: usize,
        ef: Option<usize>,
        filter: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let query_vec = query.as_slice()?;

        if query_vec.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dim,
                query_vec.len()
            )));
        }

        let query_vector = QueryVector::from(query_vec.to_vec());

        let filter_condition = filter
            .map(|f| dict_to_filter(&f))
            .transpose()?;

        if entry_point_ids.is_empty() {
            return self.search_with_filter(&query_vector, k, ef, filter_condition.as_ref());
        }

        let search_params = segment::types::SearchParams {
            hnsw_ef: ef,
            exact: false,
            quantization: None,
            indexed_only: false,
            acorn: None,
        };

        let hw_acc = HwMeasurementAcc::disposable();
        let query_context = QueryContext::new(10000, hw_acc);
        let segment_query_context = query_context.get_segment_query_context();

        let id_tracker = self.segment.id_tracker.borrow();

        // Resolve external entry point IDs to internal PointOffsetType
        let internal_entries: Vec<common::types::PointOffsetType> = entry_point_ids
            .iter()
            .filter_map(|&ext_id| {
                id_tracker.internal_id(PointIdType::NumId(ext_id))
            })
            .collect();

        // Resolve search_set external IDs to internal IDs
        let internal_search_set: Vec<common::types::PointOffsetType> = search_set_ids
            .as_ref()
            .map(|ids| {
                ids.iter()
                    .filter_map(|&ext_id| id_tracker.internal_id(PointIdType::NumId(ext_id)))
                    .collect()
            })
            .unwrap_or_default();

        drop(id_tracker);

        if internal_entries.is_empty() {
            return self.search_with_filter(&query_vector, k, ef, filter_condition.as_ref());
        }

        // Access the HNSW index directly for multi-entry search
        let vector_data = self.segment.vector_data.get(DEFAULT_VECTOR_NAME)
            .ok_or_else(|| PyValueError::new_err("Vector data not found"))?;
        let vector_index = vector_data.vector_index.borrow();

        use segment::index::vector_index_base::VectorIndexEnum;
        let vector_query_context = segment_query_context.get_vector_context(DEFAULT_VECTOR_NAME);
        match &*vector_index {
            VectorIndexEnum::Hnsw(hnsw_index) => {
                let results = hnsw_index.search_multi_entry_with_graph(
                    &query_vector,
                    filter_condition.as_ref(),
                    k,
                    Some(&search_params),
                    &internal_entries,
                    &internal_search_set,
                    &vector_query_context,
                ).map_err(|e| PyValueError::new_err(format!("Multi-entry search failed: {}", e)))?;

                drop(vector_index);
                let id_tracker = self.segment.id_tracker.borrow();
                Ok(results
                    .into_iter()
                    .filter_map(|scored| {
                        id_tracker.external_id(scored.idx).map(|ext_id| {
                            let id = match ext_id {
                                PointIdType::NumId(n) => n,
                                PointIdType::Uuid(_) => 0,
                            };
                            (id, scored.score)
                        })
                    })
                    .collect())
            }
            _ => {
                drop(vector_index);
                drop(vector_query_context);
                self.search_with_filter(&query_vector, k, ef, filter_condition.as_ref())
            }
        }
    }

    /// Get number of vectors in the segment
    fn len(&self) -> usize {
        self.segment.available_point_count()
    }

    /// Flush to disk
    fn flush(&self) -> PyResult<()> {
        self.segment
            .flush(true)
            .map(|_| ())
            .map_err(|e| PyValueError::new_err(format!("Flush failed: {}", e)))
    }
}

impl HnswSegment {
    fn search_with_filter(
        &self,
        query_vector: &QueryVector,
        k: usize,
        ef: Option<usize>,
        filter_condition: Option<&segment::types::Filter>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let search_params = segment::types::SearchParams {
            hnsw_ef: ef,
            exact: false,
            quantization: None,
            indexed_only: false,
            acorn: None,
        };
        let hw_acc = HwMeasurementAcc::disposable();
        let qc = QueryContext::new(10000, hw_acc);
        let sqc = qc.get_segment_query_context();
        let results = self.segment
            .search_batch(
                DEFAULT_VECTOR_NAME,
                &[query_vector],
                &segment::types::WithPayload::default(),
                &segment::types::WithVector::Bool(false),
                filter_condition,
                k,
                Some(&search_params),
                &sqc,
            )
            .map_err(|e| PyValueError::new_err(format!("Search failed: {}", e)))?;
        let batch = results.into_iter().next().unwrap_or_default();
        Ok(batch.into_iter().map(|sp| {
            let id = match sp.id {
                PointIdType::NumId(n) => n,
                PointIdType::Uuid(_) => 0,
            };
            (id, sp.score)
        }).collect())
    }
}

/// Convert Python dict to Qdrant payload
fn dict_to_payload(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Map<String, serde_json::Value>> {
    let mut map = serde_json::Map::new();
    
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let json_value = python_to_json(&value)?;
        map.insert(key_str, json_value);
    }
    
    Ok(map)
}

/// Convert Python object to JSON value
fn python_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(f).unwrap_or(0.into())
        ))
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(l) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in l.iter() {
            arr.push(python_to_json(&item)?);
        }
        Ok(serde_json::Value::Array(arr))
    } else {
        Ok(serde_json::Value::Null)
    }
}

/// Convert JSON map to Python dict
fn json_to_python_dict(py: Python<'_>, map: &serde_json::Map<String, serde_json::Value>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (k, v) in map {
        dict.set_item(k, json_to_python(py, v)?)?;
    }
    Ok(dict.into())
}

/// Convert JSON value to Python object
fn json_to_python(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    match v {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None()) // Fallback
            }
        },
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
           let list = PyList::empty_bound(py);
           for item in arr {
               list.append(json_to_python(py, item)?)?;
           }
           Ok(list.into())
        },
        serde_json::Value::Object(map) => json_to_python_dict(py, map),
    }
}


/// Convert Python dict to Qdrant filter
fn dict_to_filter(dict: &Bound<'_, PyDict>) -> PyResult<segment::types::Filter> {
    let mut must_conditions = Vec::new();
    
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let key_path = PayloadKeyType::from_str(&key_str)
            .map_err(|_| PyValueError::new_err("Invalid field path"))?;

        let condition = if let Ok(s) = value.extract::<String>() {
            segment::types::FieldCondition::new_match(
                key_path, 
                segment::types::Match::Value(segment::types::MatchValue { value: segment::types::ValueVariants::String(s) })
            )
        } else if let Ok(i) = value.extract::<i64>() {
            segment::types::FieldCondition::new_match(
                key_path, 
                segment::types::Match::Value(segment::types::MatchValue { value: segment::types::ValueVariants::Integer(i) })
            )
        } else if let Ok(b) = value.extract::<bool>() {
           segment::types::FieldCondition::new_match(
                key_path, 
                segment::types::Match::Value(segment::types::MatchValue { value: segment::types::ValueVariants::Bool(b) })
            )
        } else {
             continue; 
        };
        
        must_conditions.push(segment::types::Condition::Field(condition));
    }
    
    Ok(segment::types::Filter {
        should: None,
        min_should: None, 
        must: Some(must_conditions),
        must_not: None,
    })
}

#[pymodule]
fn latence_hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HnswSegment>()?;
    Ok(())
}
