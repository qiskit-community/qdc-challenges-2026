"""
Serverless pipeline steps for the glycine EWF+SQD demo.

Mirrors the proteasome_ligand_challenge steps/ package but wired for a
single-molecule (glycine) run that uses the QFWorkflow + adaptive solver
rules from the local notebook demo.

    run_meanfield   →  step 1: RHF on glycine
    run_fragments   →  step 2: Vayesta EWF fragmentation (IAO, MP2 bath)
    run_solve       →  step 3: per-fragment FCI / CCSD / SQD
    run_reconstruct →  step 4: partitioned cumulant energy reconstruction
"""

from .stage1_spec import (
    MoleculeSpec,
    HFConfig,
    Stage1Spec,
    Stage1SpecError,
    Stage1ValidationError,
    validate_scf_state,
    spec_from_pyscf,
)
from .stage2_spec import (
    FragmentationConfig,
    BathConfig,
    EWFConfig,
    Stage2Spec,
    Stage2SpecError,
    Stage2ValidationError,
    validate_ewf_dumpfile,
    ewf_diagnostics,
    build_stage2_result_record,
)
from .step1_meanfield import run_meanfield
from .step2_fragments import run_fragments
from .step3_solve import run_solve
from .step4_reconstruct import (
    run_reconstruct,
    run_reconstruct_from_refs,
    QuantumReconstructionError,
    ClassicalReconstructionError,
)
from .fragment_result_store import (
    FragmentResultReference,
    FragmentStoreError,
    save_fragment_result,
    load_fragment_result,
    load_fragment_result_from_ref,
    compute_payload_digest,
)
from .artifact_store import (
    ArtifactReference,
    ArtifactStoreError,
    ArtifactConflict,
    ArtifactNotFound,
    ArtifactCorrupt,
    LocalArtifactStore,
    KIND_MEANFIELD,
    KIND_EMBEDDING,
    KIND_FRAGMENT_H5,
    KIND_SOLVER_PKL,
    KIND_SUMMARY_JSON,
)
from .run_manifests import (
    RunManifest,
    FragmentArtifactManifest,
    ManifestError,
    RUN_MANIFEST_SCHEMA,
    FRAGMENT_MANIFEST_SCHEMA,
)
from .stage_artifacts import (
    publish_meanfield_artifact,
    retrieve_meanfield_artifact,
    publish_embedding_artifact,
    retrieve_embedding_artifact,
    publish_solver_result_artifact,
    retrieve_solver_result_artifact,
    StageArtifactError,
)
from .classical_fragment_runner import (
    run_classical_fragment,
    load_fragment_hamiltonian,
    ClassicalFragmentResult,
    ClassicalFragmentError,
    CLASSICAL_SOLVERS,
)
from .quantum_submission import (
    SubmissionRecord,
    QuantumSubmissionError,
    AmbiguousSubmissionError,
    DuplicateSubmissionError,
    QpuBudgetError,
    submit_quantum_fragment,
    build_submission_tags,
    reconcile_submission,
    resolve_ambiguous_submission,
    load_fragment_hamiltonian_for_qpu,
)
from .quantum_postprocess import (
    QuantumPostprocessResult,
    CountsArtifact,
    QuantumPostprocessError,
    MeasurementAuthenticationError,
    postprocess_quantum_fragment,
    retrieve_and_store_counts,
    verify_measurement_identity,
    POSTPROCESS_SCHEMA,
)
from .quantum_monitor import (
    QuantumStatusTable,
    MonitoringSnapshot,
    RefreshError,
    ProviderStatus,
    refresh_quantum_jobs,
    build_quantum_status_table,
    create_dispatch_record,
    list_dispatch_eligible,
    normalize_provider_status,
    MONITORING_SCHEMA,
)
from .reconstruction_coordinator import (
    BarrierReport,
    ReconstructionSummary,
    validate_completion_barrier,
    reconstruct_from_registry,
    reconstruct_from_manifest,
    RECONSTRUCTION_SCHEMA,
    BARRIER_SCHEMA,
)
from .artifact_transport import (
    TRANSPORT_SCHEMA,
    MODE_FILE_API,
    MODE_INLINE,
    TransportError,
    TransportChecksumError,
    TransportIdentityError,
    TransportNotFoundError,
    TransportDescriptor,
    input_transport_name,
    result_transport_name,
    MockServerlessFileClient,
    ServerlessFileApiClient,
    ServerlessTransportAdapter,
    build_worker_file_client,
    validate_fragment_result_file,
    verify_and_import_fragment_result,
)

__all__ = [
    # Steps
    "MoleculeSpec",
    "HFConfig",
    "Stage1Spec",
    "Stage1SpecError",
    "Stage1ValidationError",
    "validate_scf_state",
    "spec_from_pyscf",
    "run_meanfield",
    "run_fragments",
    "run_solve",
    "run_reconstruct",
    "run_reconstruct_from_refs",
    "QuantumReconstructionError",
    "ClassicalReconstructionError",
    # Fragment result store
    "FragmentResultReference",
    "FragmentStoreError",
    "save_fragment_result",
    "load_fragment_result",
    "load_fragment_result_from_ref",
    "compute_payload_digest",
    # Artifact store
    "ArtifactReference",
    "ArtifactStoreError",
    "ArtifactConflict",
    "ArtifactNotFound",
    "ArtifactCorrupt",
    "LocalArtifactStore",
    "KIND_MEANFIELD",
    "KIND_EMBEDDING",
    "KIND_FRAGMENT_H5",
    "KIND_SOLVER_PKL",
    "KIND_SUMMARY_JSON",
    # Run manifests
    "RunManifest",
    "FragmentArtifactManifest",
    "ManifestError",
    "RUN_MANIFEST_SCHEMA",
    "FRAGMENT_MANIFEST_SCHEMA",
    # Stage artifacts
    "publish_meanfield_artifact",
    "retrieve_meanfield_artifact",
    "publish_embedding_artifact",
    "retrieve_embedding_artifact",
    "publish_solver_result_artifact",
    "retrieve_solver_result_artifact",
    "StageArtifactError",
    # Classical fragment runner
    "run_classical_fragment",
    "load_fragment_hamiltonian",
    "ClassicalFragmentResult",
    "ClassicalFragmentError",
    "CLASSICAL_SOLVERS",
    # Quantum submission (3Q-A)
    "SubmissionRecord",
    "QuantumSubmissionError",
    "AmbiguousSubmissionError",
    "DuplicateSubmissionError",
    "QpuBudgetError",
    "submit_quantum_fragment",
    "build_submission_tags",
    "reconcile_submission",
    "resolve_ambiguous_submission",
    "load_fragment_hamiltonian_for_qpu",
    # Quantum post-processing (3Q-C)
    "QuantumPostprocessResult",
    "CountsArtifact",
    "QuantumPostprocessError",
    "MeasurementAuthenticationError",
    "postprocess_quantum_fragment",
    "retrieve_and_store_counts",
    "verify_measurement_identity",
    "POSTPROCESS_SCHEMA",
    # Quantum monitoring (3Q-B)
    "QuantumStatusTable",
    "MonitoringSnapshot",
    "RefreshError",
    "ProviderStatus",
    "refresh_quantum_jobs",
    "build_quantum_status_table",
    "create_dispatch_record",
    "list_dispatch_eligible",
    "normalize_provider_status",
    "MONITORING_SCHEMA",
    # Reconstruction coordinator (Stage 4 / Task 5.8)
    "BarrierReport",
    "ReconstructionSummary",
    "validate_completion_barrier",
    "reconstruct_from_registry",
    "reconstruct_from_manifest",
    "RECONSTRUCTION_SCHEMA",
    "BARRIER_SCHEMA",
    # Artifact transport (Task 5.9C.1)
    "TRANSPORT_SCHEMA",
    "MODE_FILE_API",
    "MODE_INLINE",
    "TransportError",
    "TransportChecksumError",
    "TransportIdentityError",
    "TransportNotFoundError",
    "TransportDescriptor",
    "input_transport_name",
    "result_transport_name",
    "MockServerlessFileClient",
    "ServerlessFileApiClient",
    "ServerlessTransportAdapter",
    "build_worker_file_client",
    "validate_fragment_result_file",
    "verify_and_import_fragment_result",
]
