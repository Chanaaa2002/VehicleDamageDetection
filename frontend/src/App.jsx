import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:8000`;

const FALLBACK_OPTIONS = {
  vehicles: [
    { brand: "Honda", model: "Vezel", default_year: 2015 },
    { brand: "Toyota", model: "Aqua", default_year: 2015 },
    { brand: "Toyota", model: "Vitz", default_year: 2016 },
    { brand: "Suzuki", model: "Alto", default_year: 2018 },
    { brand: "Honda", model: "Fit", default_year: 2015 },
    { brand: "Toyota", model: "Prius", default_year: 2013 },
    { brand: "Suzuki", model: "Wagon R", default_year: 2017 },
  ],
  damage_types: ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"],
  severity_levels: ["minor", "moderate", "severe"],
  damage_parts: [
    "front bumper",
    "rear bumper",
    "windshield",
    "lamp/light",
    "rear lamp/light",
    "headlight/front lamp",
    "wheel/tire",
    "hood/front shell",
    "tailgate",
    "fender/body panel",
    "front body panel",
    "rear body panel",
    "body shell",
    "internal hidden components",
  ],
};

function text(value) {
  if (value === null || value === undefined || value === "") return "Unknown";
  return String(value).replaceAll("_", " ");
}

function replacementText(value) {
  if (value === true) return "Required";
  if (value === false) return "Not required";
  return "Needs inspection";
}

function makeId() {
  return window.crypto?.randomUUID
    ? window.crypto.randomUUID()
    : `${Date.now()}-${Math.random()}`;
}

function validationTitle(item) {
  const reasonCode = item?.input_validation?.reason_code || item?.next_step || "";

  if (reasonCode === "not_vehicle_damage_image") {
    return "Not a vehicle damage image";
  }

  if (reasonCode === "vehicle_part_not_confirmed") {
    return "Vehicle not confirmed";
  }

  if (reasonCode === "no_clear_damage_detected") {
    return "No clear damage detected";
  }

  if (reasonCode === "vehicle_no_damage") {
    return "No reliable damage identified";
  }

  return "Image validation failed";
}

function PartBreakdown({ title, estimate }) {
  const parts = estimate?.matched_parts || [];
  const missing = estimate?.missing_parts || [];

  return (
    <details className="accordion">
      <summary>{title}</summary>

      <div className="total-row">
        <span>Total range</span>
        <strong>{estimate?.total_range || "Not available"}</strong>
      </div>

      {parts.map((part, index) => (
        <div className="part-card" key={`${part.part}-${index}`}>
          <div className="part-title">
            <strong>
              {text(part.detected_part)} → {text(part.part)}
            </strong>
            <span>{part.total_range}</span>
          </div>

          <p>{part.part_action}</p>

          <div className="part-tags">
            <span>Condition: {text(part.condition)}</span>
            <span>Replacement: {replacementText(part.replacement_required)}</span>
          </div>
        </div>
      ))}

      {missing.length > 0 && (
        <div className="warning-box">
          Manual quotation needed for: {missing.join(", ")}
        </div>
      )}
    </details>
  );
}

function PartsPicker({ options, selected, onToggle }) {
  return (
    <div className="parts-picker">
      {options.map((part) => (
        <button
          type="button"
          key={part}
          className={selected?.includes(part) ? "selected" : ""}
          onClick={() => onToggle(part)}
        >
          {text(part)}
        </button>
      ))}
    </div>
  );
}

function ConfirmCard({
  item,
  index,
  options,
  vehicle,
  onUpdateDraft,
  onEstimateConfirmed,
  estimating,
}) {
  if (item.error) {
    return (
      <section className="result-card validation-result-card">
        <div className="validation-card-head">
          <span>Image validation • Image {index + 1}</span>
          <h3>{validationTitle(item)}</h3>
        </div>

        <div className="warning-box validation-warning">
          {item.error}
        </div>

        <p className="validation-help">
          This image was stopped before repair-cost estimation. Please upload a clear photo of a damaged vehicle area.
        </p>
      </section>
    );
  }

  const fused = item.final_fused_result || {};
  const status = item.confirmation_status || {};
  const draft = item.confirmDraft;
  const estimate = item.confirmedEstimate;

  const confirmedCost = estimate?.cost_estimation;
  const full = confirmedCost?.possible_full_repair_estimate;
  const primary = confirmedCost?.primary_estimate;
  const repair = estimate?.repair_recommendation;

  function update(field, value) {
    onUpdateDraft(index, {
      ...draft,
      [field]: value,
    });
  }

  function togglePart(part) {
    const current = draft.possible_affected_parts || [];
    const exists = current.includes(part);

    const next = exists
      ? current.filter((item) => item !== part)
      : [...current, part];

    update("possible_affected_parts", next);
  }

  return (
    <section className="result-card">
      <div className="result-head">
        <div>
          <span>AI review • Image {index + 1}</span>
          <h3>{item.filename}</h3>
        </div>

        <b className={status.required ? "pill warn" : "pill ok"}>
          {status.required ? "Confirm" : "Accept/Edit"}
        </b>
      </div>

      <div className="ai-summary">
        <div>
          <span>Damage</span>
          <strong>{text(fused.damage_type)}</strong>
        </div>
        <div>
          <span>Severity</span>
          <strong>{text(fused.severity)}</strong>
        </div>
        <div>
          <span>Main part</span>
          <strong>{text(fused.damaged_part)}</strong>
        </div>
      </div>

      {status.reasons?.length > 0 && (
        <div className="notice-box">
          <strong>Confirmation needed</strong>
          {status.reasons.slice(0, 3).map((reason, reasonIndex) => (
            <p key={reasonIndex}>• {reason}</p>
          ))}
        </div>
      )}

      <div className="edit-panel">
        <div className="edit-title">
          <span>Step 2</span>
          <h4>Confirm before cost</h4>
        </div>

        <div className="edit-grid">
          <label>
            Damage type
            <select value={draft.damage_type} onChange={(e) => update("damage_type", e.target.value)}>
              {options.damage_types.map((type) => (
                <option key={type}>{type}</option>
              ))}
            </select>
          </label>

          <label>
            Severity
            <select value={draft.severity} onChange={(e) => update("severity", e.target.value)}>
              {options.severity_levels.map((severity) => (
                <option key={severity}>{severity}</option>
              ))}
            </select>
          </label>

          <label className="wide">
            Main damaged part
            <select value={draft.damaged_part} onChange={(e) => update("damaged_part", e.target.value)}>
              {options.damage_parts.map((part) => (
                <option key={part}>{part}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="parts-edit">
          <div className="parts-edit-head">
            <strong>Confirmed affected parts</strong>
            <small>Tap to add/remove</small>
          </div>

          <PartsPicker
            options={options.damage_parts}
            selected={draft.possible_affected_parts || []}
            onToggle={togglePart}
          />
        </div>

        <label className="check-line">
          <input
            type="checkbox"
            checked={draft.force_manual_inspection}
            onChange={(e) => update("force_manual_inspection", e.target.checked)}
          />
          Add manual inspection warning
        </label>

        <button
          className="estimate-btn"
          type="button"
          disabled={estimating}
          onClick={() => onEstimateConfirmed(index, vehicle)}
        >
          {estimating ? "Generating estimate..." : "Generate Final Cost"}
        </button>
      </div>

      {estimate && (
        <div className="final-report">
          <div className="cost-hero">
            <span>Final confirmed estimate</span>
            <strong>{full?.total_range || primary?.total_range || "Not available"}</strong>
            <p>Based on the affected parts confirmed above.</p>
          </div>

          <div className="main-cost">
            <span>Main part only</span>
            <strong>{primary?.total_range || "Not available"}</strong>
          </div>

          <div className="decision-box">
            <span>Repair guidance</span>
            <h4>{text(repair?.repair_category)}</h4>
            <p>{repair?.reason}</p>
          </div>

          <PartBreakdown title="Selected parts cost details" estimate={full} />
          <PartBreakdown title="Main part details" estimate={primary} />

          <details className="accordion">
            <summary>Important notes</summary>
            {(confirmedCost?.cost_estimation_note || []).map((note, i) => (
              <p className="note" key={i}>
                {note}
              </p>
            ))}
          </details>
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [options, setOptions] = useState(FALLBACK_OPTIONS);

  const [brand, setBrand] = useState("Honda");
  const [model, setModel] = useState("Vezel");
  const [year, setYear] = useState("2015");

  const [images, setImages] = useState([]);
  const [analysisResult, setAnalysisResult] = useState(null);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [estimatingIndex, setEstimatingIndex] = useState(null);

  const [cameraOpen, setCameraOpen] = useState(false);
  const [stream, setStream] = useState(null);

  const videoRef = useRef(null);
  const uploadRef = useRef(null);
  const fallbackCameraRef = useRef(null);

  const brands = useMemo(() => {
    return [...new Set(options.vehicles.map((vehicle) => vehicle.brand))];
  }, [options]);

  const modelsForBrand = useMemo(() => {
    return options.vehicles.filter((vehicle) => vehicle.brand === brand);
  }, [brand, options]);

  useEffect(() => {
    async function loadOptions() {
      try {
        const response = await fetch(`${API_BASE_URL}/options`);
        const data = await response.json();
        if (response.ok) setOptions(data);
      } catch {
        setOptions(FALLBACK_OPTIONS);
      }
    }

    loadOptions();
  }, []);

  useEffect(() => {
    if (modelsForBrand.length && !modelsForBrand.some((item) => item.model === model)) {
      setModel(modelsForBrand[0].model);
      setYear(String(modelsForBrand[0].default_year));
    }
  }, [brand, modelsForBrand, model]);

  useEffect(() => {
    if (cameraOpen && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }
  }, [cameraOpen, stream]);

  useEffect(() => {
    return () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, [stream]);

  function updateModel(nextModel) {
    setModel(nextModel);
    const vehicle = options.vehicles.find((item) => item.brand === brand && item.model === nextModel);
    if (vehicle) setYear(String(vehicle.default_year));
  }

  function addFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file.type.startsWith("image/"));

    const prepared = files.map((file) => ({
      id: makeId(),
      file,
      preview: URL.createObjectURL(file),
    }));

    setImages((prev) => [...prev, ...prepared]);
    setAnalysisResult(null);
    setError("");
  }

  function removeImage(id) {
    setImages((prev) => {
      const target = prev.find((item) => item.id === id);
      if (target) URL.revokeObjectURL(target.preview);
      return prev.filter((item) => item.id !== id);
    });
  }

  function clearImages() {
    images.forEach((item) => URL.revokeObjectURL(item.preview));
    setImages([]);
    setAnalysisResult(null);
    setError("");
  }

  async function openCamera() {
    setError("");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      fallbackCameraRef.current?.click();
      return;
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });

      setStream(mediaStream);
      setCameraOpen(true);
    } catch {
      setError("Camera needs HTTPS on mobile. Use deployed link or upload photos.");
      fallbackCameraRef.current?.click();
    }
  }

  function closeCamera() {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    setStream(null);
    setCameraOpen(false);
  }

  function capturePhoto() {
    const video = videoRef.current;

    if (!video || !video.videoWidth) {
      setError("Camera is not ready. Try again.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError("Could not capture photo.");
          return;
        }

        const file = new File([blob], `damage-photo-${Date.now()}.jpg`, {
          type: "image/jpeg",
        });

        addFiles([file]);
        closeCamera();
      },
      "image/jpeg",
      0.92
    );
  }

  function buildDraftFromAI(item) {
    const fused = item.final_fused_result || {};

    return {
      damage_type: fused.damage_type || "dent",
      severity: fused.severity || "minor",
      damaged_part: fused.damaged_part || "front bumper",
      possible_affected_parts: fused.possible_affected_parts || [],
      force_manual_inspection: Boolean(item.confirmation_status?.required),
    };
  }

  async function analyzeImages() {
    setError("");
    setAnalysisResult(null);

    if (!images.length) {
      setError("Take or upload at least one damage photo.");
      return;
    }

    if (!year || Number(year) < 1990 || Number(year) > 2035) {
      setError("Enter a valid vehicle year.");
      return;
    }

    const formData = new FormData();
    formData.append("brand", brand);
    formData.append("model", model);
    formData.append("year", year);

    images.forEach((image) => formData.append("files", image.file));

    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || "Analysis failed.");

      const resultsWithDrafts = data.results.map((item) => ({
        ...item,
        confirmDraft: item.error ? null : buildDraftFromAI(item),
        confirmedEstimate: null,
      }));

      setAnalysisResult({
        ...data,
        results: resultsWithDrafts,
      });
    } catch (err) {
      setError(err.message || "Could not connect to backend.");
    } finally {
      setLoading(false);
    }
  }

  function updateDraft(index, draft) {
    setAnalysisResult((prev) => {
      const nextResults = [...prev.results];
      nextResults[index] = {
        ...nextResults[index],
        confirmDraft: draft,
        confirmedEstimate: null,
      };

      return {
        ...prev,
        results: nextResults,
      };
    });
  }

  async function estimateConfirmed(index, vehicle) {
    const item = analysisResult.results[index];
    const draft = item.confirmDraft;

    try {
      setEstimatingIndex(index);

      const response = await fetch(`${API_BASE_URL}/estimate-confirmed`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          brand: vehicle.brand,
          model: vehicle.model,
          year: Number(vehicle.year),
          ...draft,
        }),
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || "Final estimate failed.");

      setAnalysisResult((prev) => {
        const nextResults = [...prev.results];
        nextResults[index] = {
          ...nextResults[index],
          confirmedEstimate: data,
        };

        return {
          ...prev,
          results: nextResults,
        };
      });
    } catch (err) {
      setError(err.message || "Could not generate final estimate.");
    } finally {
      setEstimatingIndex(null);
    }
  }

  const imageCountText = images.length ? `${images.length} photo(s)` : "No photos";

  const hasValidAnalysisResults =
    analysisResult?.results?.some((item) => !item.error) || false;

  const allAnalysisResultsInvalid =
    analysisResult?.results?.length > 0 && !hasValidAnalysisResults;

  return (
    <main className="mobile-app">
      <section className="hero">
        <div className="scan-line"></div>

        <div className="top-row">
          <div className="logo">VD</div>
          <div>
            <h1>Vehicle Damage AI</h1>
            <p>Damage detection & repair estimate</p>
          </div>
        </div>

        <h2>Inspect damage in 3 steps.</h2>
        <p className="subtext">
          Upload photos, confirm AI findings, then generate a repair cost range.
        </p>

        <div className="step-strip">
          <span>Analyze</span>
          <span>Confirm</span>
          <span>Estimate</span>
        </div>
      </section>

      <section className="card">
        <h3>Vehicle details</h3>

        <label>
          Brand
          <select value={brand} onChange={(event) => setBrand(event.target.value)}>
            {brands.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </label>

        <label>
          Model
          <select value={model} onChange={(event) => updateModel(event.target.value)}>
            {modelsForBrand.map((vehicle) => (
              <option key={vehicle.model}>{vehicle.model}</option>
            ))}
          </select>
        </label>

        <label>
          Year
          <input
            type="number"
            value={year}
            min="1990"
            max="2035"
            onChange={(event) => setYear(event.target.value)}
          />
        </label>
      </section>

      <section className="card">
        <div className="card-title-row">
          <h3>Damage photos</h3>
          <span>{imageCountText}</span>
        </div>

        <div className="action-grid">
          <button type="button" onClick={openCamera}>Camera</button>
          <button type="button" className="secondary" onClick={() => uploadRef.current.click()}>
            Upload
          </button>
        </div>

        <input
          ref={uploadRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(event) => {
            addFiles(event.target.files);
            event.target.value = "";
          }}
        />

        <input
          ref={fallbackCameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={(event) => {
            addFiles(event.target.files);
            event.target.value = "";
          }}
        />

        {images.length > 0 && (
          <>
            <div className="preview-header">
              <strong>Selected photos</strong>
              <button type="button" onClick={clearImages}>Clear</button>
            </div>

            <div className="preview-grid">
              {images.map((image) => (
                <div className="preview" key={image.id}>
                  <img src={image.preview} alt={image.file.name} />
                  <button type="button" onClick={() => removeImage(image.id)}>
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

        {error && <div className="error">{error}</div>}

        <button className="analyze-btn" type="button" onClick={analyzeImages} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze with AI"}
        </button>
      </section>

      {cameraOpen && (
        <section className="camera-modal">
          <div className="camera-box">
            <video ref={videoRef} autoPlay playsInline muted></video>

            <div className="camera-actions">
              <button type="button" onClick={capturePhoto}>Capture</button>
              <button type="button" className="cancel" onClick={closeCamera}>Close</button>
            </div>
          </div>
        </section>
      )}

      {loading && (
        <section className="card center">
          <div className="loader"></div>
          <h3>Analyzing image</h3>
          <p>Checking vehicle evidence, damage evidence, severity, and affected parts.</p>
        </section>
      )}

      {!loading && !analysisResult && (
        <section className="card center muted-card">
          <h3>Ready for inspection</h3>
          <p>Upload vehicle damage photos to start.</p>
        </section>
      )}

      {analysisResult && (
        <section className="results">
          {hasValidAnalysisResults && (
            <div className="vehicle-banner">
              <span>Selected vehicle</span>
              <strong>
                {analysisResult.vehicle.brand} {analysisResult.vehicle.model} {analysisResult.vehicle.year}
              </strong>
              <p>{analysisResult.image_count} image(s) analyzed</p>
            </div>
          )}

          {allAnalysisResultsInvalid && (
            <div className="validation-banner">
              <span>Image validation</span>
              <strong>No valid vehicle damage detected</strong>
              <p>Please upload a clear photo of a damaged vehicle area.</p>
            </div>
          )}

          {analysisResult.results.map((item, index) => (
            <ConfirmCard
              key={`${item.filename}-${index}`}
              item={item}
              index={index}
              options={options}
              vehicle={analysisResult.vehicle}
              onUpdateDraft={updateDraft}
              onEstimateConfirmed={estimateConfirmed}
              estimating={estimatingIndex === index}
            />
          ))}
        </section>
      )}
    </main>
  );
}