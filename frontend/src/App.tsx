import React, { useEffect, useState } from "react";

type PredictionResult = {
  label: string;
  confidence: number;
};

type ImageItem = {
  file: File;
  previewUrl: string;
  prediction: PredictionResult | null;
  error: string | null;
};

const API_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000/predict";

const App: React.FC = () => {
  const [items, setItems] = useState<ImageItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Clean up object URLs when component unmounts or items change
  useEffect(() => {
    return () => {
      items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    };
  }, [items]);

  const resetState = () => {
    setGlobalError(null);
  };

  const buildItemsFromFiles = (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const imageFiles = fileArray.filter((f) => f.type.startsWith("image/"));

    if (imageFiles.length === 0) {
      setGlobalError("Please upload at least one valid image file.");
      setItems([]);
      return;
    }

    const newItems: ImageItem[] = imageFiles.map((file) => ({
      file,
      previewUrl: URL.createObjectURL(file),
      prediction: null,
      error: null,
    }));

    // Clean up previews from previous selection
    items.forEach((item) => URL.revokeObjectURL(item.previewUrl));

    resetState();
    setItems(newItems);
  };

  const onFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    buildItemsFromFiles(event.target.files);
  };

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);

    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      buildItemsFromFiles(event.dataTransfer.files);
    }
  };

  const onDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!isDragging) {
      setIsDragging(true);
    }
  };

  const onDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleClear = () => {
    items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    setItems([]);
    setGlobalError(null);
  };

  const handleSubmit = async () => {
    if (items.length === 0) {
      setGlobalError("Please select at least one image before predicting.");
      return;
    }

    setLoading(true);
    setGlobalError(null);

    setItems((prev) =>
      prev.map((it) => ({ ...it, prediction: null, error: null }))
    );

    try {
      const updatedItems: ImageItem[] = [];

      for (const item of items) {
        const formData = new FormData();
        formData.append("file", item.file);

        try {
          const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const errJson = await response.json().catch(() => null);
            const message =
              errJson?.error ||
              `Prediction failed for ${item.file.name} (status ${response.status}).`;

            updatedItems.push({
              ...item,
              prediction: null,
              error: message,
            });
            continue;
          }

          const data = (await response.json()) as PredictionResult;
          updatedItems.push({
            ...item,
            prediction: data,
            error: null,
          });
        } catch (err) {
          console.error(err);
          updatedItems.push({
            ...item,
            prediction: null,
            error: `Error predicting for ${item.file.name}.`,
          });
        }
      }

      setItems(updatedItems);
    } catch (err) {
      console.error(err);
      setGlobalError(
        err instanceof Error
          ? err.message
          : "Something went wrong while calling the API."
      );
    } finally {
      setLoading(false);
    }
  };

  const firstPreview = items.length > 0 ? items[0].previewUrl : null;

  return (
    <div className="app-root">
      <div className="card">
        <header className="card-header">
          <h1>Car Damage Classifier</h1>
          <p>
            Upload one or multiple car images and get a damage prediction for
            each.
          </p>
        </header>

        <div
          className={`drop-zone ${isDragging ? "drop-zone--active" : ""} ${
            firstPreview ? "drop-zone--has-image" : ""
          }`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          {firstPreview ? (
            <div className="preview-wrapper">
              <img
                src={firstPreview}
                alt="Preview"
                className="preview-image"
              />
            </div>
          ) : (
            <div className="drop-zone-content">
              <p className="drop-zone-title">Drag &amp; drop images here</p>
              <p className="drop-zone-subtitle">
                or click to browse (you can select multiple files)
              </p>
            </div>
          )}

          <input
            type="file"
            accept="image/*"
            multiple
            className="file-input"
            onChange={onFileInputChange}
          />
        </div>

        {items.length > 1 && (
          <p className="multi-count">
            {items.length} images selected. The first one is shown above.
          </p>
        )}

        <div className="card-actions">
          <button
            className="btn btn-secondary"
            type="button"
            onClick={handleClear}
            disabled={loading || items.length === 0}
          >
            Clear
          </button>
          <button
            className="btn btn-primary"
            type="button"
            onClick={handleSubmit}
            disabled={items.length === 0 || loading}
          >
            {loading ? "Analyzing..." : "Predict for all"}
          </button>
        </div>

        {loading && (
          <div className="status status--info">
            <span className="spinner" aria-hidden="true" />
            <span>Running the model on all selected images...</span>
          </div>
        )}

        {globalError && (
          <div className="status status--error">
            <strong>Error:</strong> {globalError}
          </div>
        )}

        {items.length > 0 && (
          <div className="multi-results">
            {items.map((item, idx) => {
              const confPercent =
                item.prediction != null
                  ? (item.prediction.confidence * 100).toFixed(1)
                  : null;

              return (
                <div key={idx} className="multi-item">
                  <div className="multi-thumb-wrapper">
                    <img
                      src={item.previewUrl}
                      alt={item.file.name}
                      className="multi-thumb"
                    />
                  </div>
                  <div className="multi-info">
                    <div className="multi-filename">{item.file.name}</div>
                    {item.error && (
                      <div className="status status--error status--compact">
                        {item.error}
                      </div>
                    )}
                    {item.prediction && !item.error && (
                      <>
                        <div className="badge">
                          <span className="badge-label">Prediction</span>
                          <span className="badge-value">
                            {item.prediction.label}
                          </span>
                        </div>
                        {confPercent && (
                          <p className="confidence">
                            Confidence:{" "}
                            <span className="confidence-value">
                              {confPercent}%
                            </span>
                          </p>
                        )}
                      </>
                    )}
                    {!item.prediction && !item.error && !loading && (
                      <p className="pending-text">
                        Prediction pending. Click{" "}
                        <strong>Predict for all</strong>.
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <footer className="card-footer">
          <small>
            Backend: FastAPI @ <code>POST /predict</code>, Model: ResNet50
          </small>
        </footer>
      </div>
    </div>
  );
};

export default App;
