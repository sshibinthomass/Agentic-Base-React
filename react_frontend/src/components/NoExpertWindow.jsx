import React, { useState, useRef } from "react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export function NoExpertWindow({
  sidebarOpen,
  onToggleSidebar,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState(null);
  const [selectedRoleIndex, setSelectedRoleIndex] = useState(0);
  const [selectedExpertIndex, setSelectedExpertIndex] = useState(0);
  const [showFullRoadmap, setShowFullRoadmap] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type !== "application/pdf") {
        setError("Please select a PDF file");
        return;
      }
      setSelectedFile(file);
      setUploadedFileName(file.name);
      setError("");
      setResults(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a PDF file first");
      return;
    }

    setProcessing(true);
    setError("");
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${BACKEND_URL}/no-expert/process`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = "Failed to process PDF";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      setResults(data.result);
      setSelectedRoleIndex(0);
      setSelectedExpertIndex(0);
    } catch (err) {
      setError(err.message || "Something went wrong while processing the PDF");
    } finally {
      setProcessing(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setUploadedFileName(null);
    setResults(null);
    setError("");
    setSelectedRoleIndex(0);
    setSelectedExpertIndex(0);
    setShowFullRoadmap(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleRoleSelect = (index) => {
    setSelectedRoleIndex(index);
    setSelectedExpertIndex(0);
  };

  const handleExpertNavigation = (direction) => {
    const currentRole = results?.roles_needed?.[selectedRoleIndex];
    const experts = currentRole?.experts_list?.enriched_scored_experts || [];
    if (direction === "next" && selectedExpertIndex < experts.length - 1) {
      setSelectedExpertIndex(selectedExpertIndex + 1);
    } else if (direction === "prev" && selectedExpertIndex > 0) {
      setSelectedExpertIndex(selectedExpertIndex - 1);
    }
  };

  const getRoleColor = (index) => {
    const colors = [
      "#3b82f6", // blue
      "#1e40af", // dark blue
      "#1e40af", // dark blue
      "#1e40af", // dark blue
      "#1e40af", // dark blue
      "#7c3aed", // purple
      "#dc2626", // red
    ];
    return colors[index % colors.length];
  };

  if (!results) {
    return (
      <section className="chat-pane">
        <div className="chat-box">
          <header className="chat-header">
            <div className="chat-header__left">
              <button
                type="button"
                className="chat-header__toggle"
                onClick={onToggleSidebar}
                aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
              >
                {sidebarOpen ? "◀" : "▶"}
              </button>
              <span className="chat-header__title">No Expert - PDF Analysis</span>
            </div>
            <button
              type="button"
              className="chat-header__clear"
              onClick={handleClear}
              disabled={processing}
            >
              Clear
            </button>
          </header>

          <div className="no-expert-content">
            <div className="no-expert-upload">
              <div className="upload-area">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={handleFileSelect}
                  className="file-input"
                  id="pdf-upload"
                  disabled={processing}
                />
                <label htmlFor="pdf-upload" className="upload-label">
                  {uploadedFileName ? (
                    <div className="file-selected">
                      <span className="file-icon">📄</span>
                      <span className="file-name">{uploadedFileName}</span>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <span className="upload-icon">📎</span>
                      <span>Click to select a PDF file</span>
                    </div>
                  )}
                </label>
              </div>

              {uploadedFileName && (
                <button
                  type="button"
                  className="process-button"
                  onClick={handleUpload}
                  disabled={processing}
                >
                  {processing ? "Processing..." : "Process PDF"}
                </button>
              )}

              {error && <div className="chat-error">{error}</div>}
            </div>

            {processing && (
              <div className="processing-overlay">
                <div className="processing-spinner"></div>
                <p>Processing PDF and identifying experts...</p>
              </div>
            )}
          </div>
        </div>
      </section>
    );
  }

  const roles = results.roles_needed || [];
  const currentRole = roles[selectedRoleIndex];
  const experts = currentRole?.experts_list?.enriched_scored_experts || [];
  const currentExpert = experts[selectedExpertIndex];
  const roadmap = results.roadmap || "";
  const roadmapPreview = showFullRoadmap || roadmap.length <= 500 
    ? roadmap 
    : roadmap.substring(0, 500);

  return (
    <section className="chat-pane">
      <div className="chat-box roadmap-layout">
        <header className="chat-header">
          <div className="chat-header__left">
            <button
              type="button"
              className="chat-header__toggle"
              onClick={onToggleSidebar}
              aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            >
              {sidebarOpen ? "◀" : "▶"}
            </button>
            <span className="chat-header__title">Roadmap</span>
          </div>
          <button
            type="button"
            className="chat-header__clear"
            onClick={handleClear}
          >
            Clear
          </button>
        </header>

        <div className="roadmap-container">
          {/* Top Section: Horizontal Tabs and Roadmap Description */}
          <div className="roadmap-top-section">
            <div className="roadmap-tabs-horizontal">
              {roles.map((role, index) => (
                <button
                  key={index}
                  className={`roadmap-tab ${selectedRoleIndex === index ? "active" : ""}`}
                  onClick={() => handleRoleSelect(index)}
                  style={{
                    backgroundColor: selectedRoleIndex === index 
                      ? getRoleColor(index) 
                      : "#f1f5f9",
                    color: selectedRoleIndex === index ? "#ffffff" : "#475569",
                  }}
                >
                  {role.broader_area}
                </button>
              ))}
            </div>

            {roadmap && (
              <div className="roadmap-description-section">
                <div className="roadmap-description">
                  {roadmapPreview}
                  {roadmap.length > 500 && (
                    <>
                      {!showFullRoadmap && "..."}
                      <button
                        className="roadmap-toggle"
                        onClick={() => setShowFullRoadmap(!showFullRoadmap)}
                      >
                        {showFullRoadmap ? (
                          <>
                            <span className="toggle-icon">▲</span> Show less
                          </>
                        ) : (
                          <>
                            <span className="toggle-icon">▼</span> Show more
                          </>
                        )}
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Bottom Section: Sidebar + Main Content */}
          <div className="roadmap-bottom-section">
            {/* Left Sidebar */}
            <div className="roadmap-sidebar">
              {roles.map((role, index) => (
                <button
                  key={index}
                  className={`roadmap-sidebar-item ${
                    selectedRoleIndex === index ? "active" : ""
                  }`}
                  onClick={() => handleRoleSelect(index)}
                  style={{
                    backgroundColor: selectedRoleIndex === index 
                      ? getRoleColor(index) + "20" 
                      : "transparent",
                    borderLeftColor: selectedRoleIndex === index 
                      ? getRoleColor(index) 
                      : "transparent",
                    color: selectedRoleIndex === index ? "#1f2937" : "#64748b",
                  }}
                >
                  {role.broader_area}
                </button>
              ))}
            </div>

            {/* Right Main Content */}
            <div className="roadmap-main-content">
              {currentRole && (
                <>
                  <div className="expert-header-section">
                    <h2 className="expert-category-title">
                      Expert: {currentRole.broader_area}
                    </h2>
                    {experts.length > 1 && (
                      <div className="expert-pagination">
                        <button
                          className="pagination-btn"
                          onClick={() => handleExpertNavigation("prev")}
                          disabled={selectedExpertIndex === 0}
                        >
                          ◀
                        </button>
                        <span className="pagination-info">
                          {selectedExpertIndex + 1}/{experts.length}
                        </span>
                        <button
                          className="pagination-btn"
                          onClick={() => handleExpertNavigation("next")}
                          disabled={selectedExpertIndex === experts.length - 1}
                        >
                          ▶
                        </button>
                      </div>
                    )}
                  </div>

                  {currentExpert && (
                    <div className="expert-details">
                      <div className="expert-profile">
                        <div className="expert-image-placeholder">
                          {currentExpert.linkedin_image ? (
                            <img
                              src={currentExpert.linkedin_image}
                              alt={`${currentExpert.first_name} ${currentExpert.last_name}`}
                            />
                          ) : (
                            <div className="no-image">No image</div>
                          )}
                        </div>
                        <div className="expert-info">
                          <h3 className="expert-name">
                            {currentExpert.first_name} {currentExpert.last_name}
                          </h3>
                          <p className="expert-role">{currentExpert.function}</p>
                          {currentExpert.source_link && (
                            <a
                              href={currentExpert.source_link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="expert-link"
                            >
                              [{currentExpert.source}]
                            </a>
                          )}
                          <p className="expert-description">
                            {currentExpert.purpose_description}
                          </p>
                        </div>
                      </div>

                      {currentRole.talking_points && (
                        <div className="talking-points-section">
                          <h4 className="section-subtitle">Talking points</h4>
                          {currentRole.talking_points.talking_points && (
                            <ul className="talking-points-list">
                              {currentRole.talking_points.talking_points.map(
                                (point, idx) => (
                                  <li key={idx}>{point}</li>
                                )
                              )}
                            </ul>
                          )}
                        </div>
                      )}

                      {currentRole.talking_points?.blocker_points && (
                        <div className="blockers-section">
                          <h4 className="section-subtitle">Blockers</h4>
                          <ul className="blockers-list">
                            {currentRole.talking_points.blocker_points.map(
                              (blocker, idx) => (
                                <li key={idx}>{blocker}</li>
                              )
                            )}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
