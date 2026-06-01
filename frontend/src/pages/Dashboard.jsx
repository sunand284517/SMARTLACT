import React, { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import api from '../utils/api';
import { Upload, Activity, Clock, FileWarning, Trash2, RefreshCw, TrendingUp, Database, CheckCircle, AlertCircle, Milk, Download, Search, Filter, Calendar, X, Lightbulb, Info, AlertTriangle, BarChart as BarChartIcon, LineChart as LineChartIcon } from 'lucide-react';
import html2pdf from 'html2pdf.js';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell 
} from 'recharts';

const Dashboard = () => {
  const { user } = useContext(AuthContext);
  const [file, setFile] = useState(null);
  const [cowId, setCowId] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Search and Filter States
  const [searchQuery, setSearchQuery] = useState('');
  const [filterClass, setFilterClass] = useState('All');
  const [filterDate, setFilterDate] = useState('');

  const fetchHistory = async () => {
    try {
      const res = await api.get('/inference/history');
      setHistory(res.data?.data || res.data || []);
    } catch (err) {
      console.error('Failed to fetch history', err);
    }
  };

  useEffect(() => {
    fetchHistory();
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('cowId', cowId);

    try {
      await api.post('/inference/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setFile(null);
      setCowId('');
      fetchHistory();
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to upload image');
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    if (!status) return <span className="badge badge-pending">pending</span>;
    const statusLower = status.toLowerCase();
    return <span className={`badge badge-${statusLower}`}>{statusLower}</span>;
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to remove this sonogram result?')) return;
    try {
      await api.delete(`/inference/${id}`);
      fetchHistory(); // Refresh visual history
    } catch (err) {
      console.error('Failed to delete history item', err);
    }
  };

  const handleDownloadPDF = (item) => {
    const element = document.createElement('div');
    element.innerHTML = `
      <div style="padding: 40px; font-family: 'Helvetica', 'Arial', sans-serif; color: #1a1a1a; position: relative; min-height: 800px;">
        <!-- Watermark Background -->
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(-30deg); opacity: 0.05; z-index: -1; pointer-events: none;">
          <img src="https://img.icons8.com/ios-filled/500/cow.png" style="width: 600px;" />
        </div>

        <!-- Header -->
        <div style="text-align: center; border-bottom: 3px solid #10B981; padding-bottom: 20px; margin-bottom: 40px;">
          <h1 style="color: #10B981; margin: 0; font-size: 32px; letter-spacing: 2px;">SMART LACT AI</h1>
          <p style="color: #666; margin: 5px 0 0 0; font-size: 14px; text-transform: uppercase;">Professional Dairy Analysis Report</p>
        </div>

        <!-- Main Content Card -->
        <div style="background: white; border: 1px solid #e1e1e1; border-radius: 12px; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
            
            <!-- Record Info -->
            <div>
              <p style="margin: 0 0 5px 0; font-size: 12px; color: #888; text-transform: uppercase;">Record Details</p>
              <h3 style="margin: 0 0 20px 0; color: #1a1a1a;">${item.cowId}</h3>
              
              <p style="margin: 0 0 5px 0; font-size: 12px; color: #888; text-transform: uppercase;">Date & Time</p>
              <p style="margin: 0 0 20px 0; font-weight: 600;">${new Date(item.createdAt).toLocaleString()}</p>
              
              <p style="margin: 0 0 5px 0; font-size: 12px; color: #888; text-transform: uppercase;">Analysis ID</p>
              <p style="margin: 0; font-family: monospace; font-size: 11px; color: #555;">${item._id}</p>
            </div>

            <!-- Analysis Results -->
            <div style="background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 5px solid #10B981;">
              <p style="margin: 0 0 15px 0; font-size: 12px; color: #888; text-transform: uppercase; font-weight: 700;">Classification Results</p>
              
              <div style="margin-bottom: 20px;">
                <span style="display: block; font-size: 24px; font-weight: 800; color: #10B981;">${item.classification || 'Processing...'}</span>
                <span style="font-size: 14px; color: #059669; font-weight: 600;">${((item.confidence || 0) * 100).toFixed(1)}% Confidence Score</span>
              </div>

              <div style="border-top: 1px solid #ddd; padding-top: 15px; margin-top: 15px;">
                <p style="margin: 0 0 5px 0; font-size: 12px; color: #888; text-transform: uppercase;">Predicted Daily Yield</p>
                <p style="margin: 0; font-size: 28px; font-weight: 700; color: #1a1a1a;">
                  ${item.yield_litres ? Number(item.yield_litres).toFixed(2) : '0.00'} <span style="font-size: 16px; color: #666; font-weight: 400;">L/day</span>
                </p>
              </div>
            </div>

          </div>
        </div>

        <!-- Footer -->
        <div style="position: absolute; bottom: 40px; left: 40px; right: 40px; text-align: center; border-top: 1px solid #eee; padding-top: 20px;">
          <p style="margin: 0; font-size: 11px; color: #aaa;">This report was automatically generated by Smart Lact AI Platform. Analysis based on deep learning sonogram echotexture classification.</p>
        </div>
      </div>
    `;

    const opt = {
      margin: [0, 0],
      filename: `Report_${item.cowId}_${new Date().getTime()}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(element).save();
  };

  const getAIInsight = (item) => {
    if (!item.classification) return null;

    const insights = [];
    const classification = item.classification.toLowerCase();
    const confidence = item.confidence || 0;
    const yield_val = item.yield_litres || 0; // ✅ FIXED KEY

    // Confidence-based insights
    if (confidence > 0.95) {
      insights.push({
        text: "Extremely high confidence prediction. Results are highly reliable for clinical decision support.",
        icon: <CheckCircle size={16} />,
        color: "var(--primary)"
      });
    } else if (confidence < 0.7) {
      insights.push({
        text: "Lower confidence detected. Consider re-scanning or manual verification for accuracy.",
        icon: <AlertTriangle size={16} />,
        color: "var(--warning)"
      });
    }

    // Classification-based insights
    if (classification.includes('peak')) {
      insights.push({
        text: "Cow is in peak production. Ensure high-energy rations and monitor for metabolic stress.",
        icon: <Lightbulb size={16} />,
        color: "var(--info)"
      });
    } else if (classification.includes('dry')) {
      insights.push({
        text: "Dry period detected. Focus on udder health and preparation for the next lactation cycle.",
        icon: <Info size={16} />,
        color: "var(--secondary)"
      });
    } else if (classification.includes('fresh')) {
      insights.push({
        text: "Fresh cow stage. Monitor closely for post-calving complications and transition health.",
        icon: <Activity size={16} />,
        color: "var(--primary)"
      });
    }

    // Yield-based insights
    if (yield_val > 0 && yield_val < 15) {
      insights.push({
        text: "Yield is below optimal levels for this stage. Check hydration and forage quality.",
        icon: <AlertCircle size={16} />,
        color: "var(--error)"
      });
    } else if (yield_val > 40) {
      insights.push({
        text: "Exceptional yield predicted. Ensure adequate mineral supplementation to support high output.",
        icon: <TrendingUp size={16} />,
        color: "var(--primary)"
      });
    }

    // Default insight if none triggered
    if (insights.length === 0) {
      insights.push({
        text: "Standard lactation profile. Continue routine management protocols.",
        icon: <Info size={16} />,
        color: "var(--text-muted)"
      });
    }

    return insights[0]; 
  };

  // Calculate stats from history safely
  const totalScans = history.length;
  const completedScans = history.filter(h => h.status && h.status.toLowerCase() === 'completed').length;
  const pendingScans = history.filter(h => h.status && h.status.toLowerCase() === 'pending').length;
  const processingScans = history.filter(h => h.status && h.status.toLowerCase() === 'processing').length;
  
  // Calculate average predicted yield safely using yield_litres ✅
  const completedWithYield = history.filter(h => h.status && h.status.toLowerCase() === 'completed' && h.yield_litres !== null && h.yield_litres !== undefined);
  const avgYield = completedWithYield.length > 0 
    ? (completedWithYield.reduce((sum, h) => sum + Number(h.yield_litres || 0), 0) / completedWithYield.length).toFixed(1)
    : '0.0';

  // Get latest completed result for display safely
  const latestResult = history.find(h => h.status && h.status.toLowerCase() === 'completed' && h.classification);

  // Get unique classifications for filter dropdown
  const uniqueClassifications = ['All', ...new Set(history.filter(h => h.classification).map(h => h.classification))];

  // --- DATA PREPARATION FOR CHARTS ---
  
  // 1. Line Chart Data: Yield over Time safely using yield_litres ✅
  const lineChartData = history
    .filter(h => h.status && h.status.toLowerCase() === 'completed' && h.yield_litres !== null && h.yield_litres !== undefined)
    .sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt))
    .map(h => ({
      date: new Date(h.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
      yield: parseFloat(Number(h.yield_litres || 0).toFixed(2)), // ✅ FIXED CRASH FALLBACK
      cowId: h.cowId
    }))
    .slice(-10); 

  // 2. Bar Chart Data: Classification Distribution
  const classCounts = history
    .filter(h => h.status && h.status.toLowerCase() === 'completed' && h.classification)
    .reduce((acc, curr) => {
      acc[curr.classification] = (acc[curr.classification] || 0) + 1;
      return acc;
    }, {});

  const barChartData = Object.entries(classCounts).map(([name, value]) => ({
    name,
    value
  }));

  const COLORS = ['#10B981', '#06B6D4', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6'];

  // Filtering Logic
  const filteredHistory = history.filter(item => {
    const matchesSearch = 
      item.cowId.toLowerCase().includes(searchQuery.toLowerCase()) || 
      (item.classification && item.classification.toLowerCase().includes(searchQuery.toLowerCase())) ||
      item._id.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesClass = filterClass === 'All' || item.classification === filterClass;
    
    const matchesDate = !filterDate || new Date(item.createdAt).toLocaleDateString() === new Date(filterDate).toLocaleDateString();

    return matchesSearch && matchesClass && matchesDate;
  });

  return (
    <div className="dashboard-page-wrapper animate-fade-in">
      {/* Premium Background Elements */}
      <div className="bg-blob blob-1"></div>
      <div className="bg-blob blob-2"></div>
      <div className="bg-blob blob-3"></div>
      
      <div className="floating-symbols">
        <div className="floating-symbol" style={{ left: '10%', animationDelay: '0s' }}>🐄</div>
        <div className="floating-symbol" style={{ left: '30%', animationDelay: '5s' }}>🥛</div>
        <div className="floating-symbol" style={{ left: '50%', animationDelay: '2s' }}>🌿</div>
        <div className="floating-symbol" style={{ left: '70%', animationDelay: '8s' }}>🧀</div>
        <div className="floating-symbol" style={{ left: '90%', animationDelay: '12s' }}>🐄</div>
      </div>

      <div className="container" style={{ position: 'relative', zIndex: 2 }}>
        {/* Dashboard Header */}
        <div className="dashboard-header">
          <div className="dashboard-title-section">
            <h1>Dairy Management Dashboard</h1>
            <p className="dashboard-subtitle">AI-powered lactation analysis and yield prediction</p>
          </div>
          <div className="dashboard-actions">
            <div className="live-indicator">
              <div className="live-dot"></div>
              <span>Live System</span>
            </div>
            <button className="btn-refresh" onClick={fetchHistory}>
              <RefreshCw size={18} /> Sync Data
            </button>
          </div>
        </div>

        {/* Stats Cards Grid */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon green">
              <Milk size={28} style={{ color: '#10B981' }} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{avgYield}</div>
              <div className="stat-label">Daily Milk Yield (L)</div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon cyan">
              <Database size={28} style={{ color: '#06B6D4' }} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{totalScans}</div>
              <div className="stat-label">Total Scans</div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon blue">
              <CheckCircle size={28} style={{ color: '#3B82F6' }} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{completedScans}</div>
              <div className="stat-label">Completed</div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon orange">
              <AlertCircle size={28} style={{ color: '#F59E0B' }} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{pendingScans + processingScans}</div>
              <div className="stat-label">Pending / Processing</div>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-2 mb-2">
          {/* Yield Trend Line Chart */}
          <div className="glass-panel" style={{ height: '400px', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ marginBottom: '1.5rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <LineChartIcon size={24} style={{ color: 'var(--primary)' }} /> 
              Yield Trends
            </h3>
            <div style={{ flex: 1, width: '100%', minHeight: '0' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lineChartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis 
                    dataKey="date" 
                    stroke="var(--text-muted)" 
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    stroke="var(--text-muted)" 
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                    unit="L"
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(26, 27, 30, 0.9)',
                      border: '1px solid var(--border)',
                      borderRadius: '8px',
                      color: 'var(--text-main)'
                    }}
                    itemStyle={{ color: 'var(--primary)' }}
                  />
                  <Legend verticalAlign="top" height={36}/>
                  <Line 
                    name="Daily Yield (L)"
                    type="monotone" 
                    dataKey="yield" 
                    stroke="var(--primary)" 
                    strokeWidth={3}
                    dot={{ fill: 'var(--primary)', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, strokeWidth: 0 }}
                    animationDuration={1500}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Classification Distribution Bar Chart */}
          <div className="glass-panel" style={{ height: '400px', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ marginBottom: '1.5rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <BarChartIcon size={24} style={{ color: 'var(--primary)' }} /> 
              Lactation Distribution
            </h3>
            <div style={{ flex: 1, width: '100%', minHeight: '0' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barChartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis 
                    dataKey="name" 
                    stroke="var(--text-muted)" 
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    stroke="var(--text-muted)" 
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip 
                    cursor={{fill: 'rgba(255,255,255,0.05)'}}
                    contentStyle={{ 
                      backgroundColor: 'rgba(26, 27, 30, 0.9)',
                      border: '1px solid var(--border)',
                      borderRadius: '8px',
                      color: 'var(--text-main)'
                    }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} name="Count" animationDuration={1500}>
                    {barChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-2 mb-2">
          {/* Left Panel - Upload */}
          <div className="glass-panel">
            <h3 style={{ marginBottom: '0.75rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Upload size={24} style={{ color: 'var(--primary)' }} /> 
              New Analysis
            </h3>
            <p className="text-muted" style={{ fontSize: '1rem', marginBottom: '1.5rem' }}>Select a cow sonogram image for AI-powered processing</p>
            
            <div style={{ padding: '1rem', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.1)', borderRadius: '12px', marginBottom: '2rem', fontSize: '0.9rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Clock size={18} />
              <span>Processing time: ~10-30 seconds</span>
            </div>
            
            {error && <div style={{ color: 'var(--error)', background: 'rgba(239,68,68,0.1)', padding: '1rem', borderRadius: '12px', marginBottom: '1.5rem', border: '1px solid rgba(239,68,68,0.2)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <FileWarning size={20} />
              <span>{error}</span>
            </div>}

            <form onSubmit={handleUpload}>
              <div className="form-group">
                <label className="form-label">Cow Identifier</label>
                <input 
                  type="text" 
                  className="form-input-modern" 
                  value={cowId} 
                  onChange={e => setCowId(e.target.value)} 
                  required 
                  placeholder="e.g. Cow #4052"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Sonogram Image</label>
                <div 
                  className={`upload-area ${file ? 'has-file' : ''}`}
                  onClick={() => document.getElementById('file-input').click()}
                >
                  <input 
                    id="file-input"
                    type="file" 
                    style={{ display: 'none' }}
                    accept="image/*" 
                    onChange={e => setFile(e.target.files[0])} 
                    required 
                  />
                  <Upload size={48} className="upload-icon" style={{ color: file ? 'var(--primary)' : 'var(--text-muted)', opacity: file ? 1 : 0.5 }} />
                  <div className="upload-text">
                    {file ? file.name : 'Drop sonogram here or click to browse'}
                  </div>
                  <div className="upload-hint">Supports PNG, JPG up to 10MB</div>
                </div>
              </div>
              
              <button type="submit" className="premium-btn-submit" style={{ width: '100%' }} disabled={isLoading || !file}>
                {isLoading ? (
                  <>
                    <div className="spinner"></div>
                    <span>Analyzing Echotexture...</span>
                  </>
                ) : (
                  <><Activity size={20} /> Analyze Sonogram</>
                )}
              </button>
            </form>
          </div>

          {/* Right Panel - Results */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ marginBottom: '0.75rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Activity size={24} style={{ color: 'var(--primary)' }} /> 
              Analysis Results
            </h3>
            <p className="text-muted" style={{ fontSize: '1rem', marginBottom: '1.5rem' }}>Insights from the neural network model</p>
            
            <div style={{ flex: 1, overflowY: 'auto', maxHeight: '500px', paddingRight: '0.5rem' }}>
              {!latestResult ? (
                <div className="text-center mt-2" style={{ textAlign: 'center', padding: '4rem 2rem', background: 'var(--bg-panel)', borderRadius: '16px', border: '1px solid var(--border)' }}>
                  <FileWarning size={48} style={{ marginBottom: '1.5rem', opacity: 0.3, color: 'var(--text-muted)' }} />
                  <p className="font-important" style={{ color: 'var(--text-main)', fontSize: '1.125rem' }}>Awaiting Analysis</p>
                  <p className="text-muted" style={{ fontSize: '0.95rem', marginTop: '0.5rem' }}>Upload a cow sonogram to generate insights</p>
                  {isLoading && (
                    <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'var(--primary-glow)', borderRadius: '16px', border: '1px solid rgba(16, 185, 129, 0.1)' }}>
                      <div className="spinner" style={{ marginBottom: '1rem', width: '2rem', height: '2rem' }}></div>
                      <p className="font-important" style={{ fontSize: '1rem', color: 'var(--primary)' }}>AI Model is thinking...</p>
                      <p className="text-muted" style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>Scanning pixels for lactation patterns</p>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  {/* Latest Result Card */}
                  <div className="result-card" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '1.5rem' }}>
                    <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <div className="result-label">Classification</div>
                        <span className={`classification-badge font-important ${latestResult.classification.toLowerCase().replace(/\s+/g, '-')}`}>
                          {latestResult.classification}
                        </span>
                      </div>
                      {latestResult.confidence !== undefined && (
                        <div style={{ textAlign: 'right' }}>
                          <div className="result-label">Confidence</div>
                          <div className="result-value font-important" style={{ color: 'var(--primary)' }}>{((latestResult.confidence || 0) * 100).toFixed(1)}%</div>
                        </div>
                      )}
                    </div>

                    {latestResult.confidence !== undefined && (
                      <div style={{ marginBottom: '1.5rem' }}>
                        <div className="progress-bar" style={{ height: '10px' }}>
                          <div 
                            className="progress-fill" 
                            style={{ width: `${(latestResult.confidence || 0) * 100}%`, background: 'var(--primary)', boxShadow: '0 0 12px var(--primary-glow)' }}
                          ></div>
                        </div>
                      </div>
                    )}

                    {/* ✅ TARGETED READ FROM yield_litres */}
                    {latestResult.yield_litres !== undefined && latestResult.yield_litres !== null && (
                      <div style={{ padding: '1.25rem', background: 'rgba(16, 185, 129, 0.05)', borderRadius: '12px', border: '1px solid rgba(16, 185, 129, 0.1)' }}>
                        <div className="result-label">Predicted Daily Milk Yield</div>
                        <div className="result-value" style={{ fontSize: '1.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <Milk size={28} style={{ color: 'var(--primary)' }} />
                          {Number(latestResult.yield_litres || 0).toFixed(2)} <span style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: '500' }}>L/day</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Recent Activity Section */}
        <div className="glass-panel" style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '2rem' }}>
            <div>
              <h3 style={{ marginBottom: '0.5rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <Clock size={24} style={{ color: 'var(--primary)' }} /> 
                Recent Activity
              </h3>
              <p className="text-muted" style={{ fontSize: '1rem' }}>Historical analysis results</p>
            </div>

            {/* Search and Filters UI */}
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', flex: 1, justifyContent: 'flex-end' }}>
              <div className="input-group" style={{ maxWidth: '300px', flex: 1 }}>
                <Search className="input-icon" size={18} />
                <input 
                  type="text" 
                  placeholder="Search Cow ID or Stage..." 
                  className="premium-input"
                  style={{ padding: '0.75rem 1rem 0.75rem 0.5rem' }}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {searchQuery && (
                  <X 
                    size={16} 
                    style={{ marginRight: '1rem', cursor: 'pointer', color: 'var(--text-muted)' }} 
                    onClick={() => setSearchQuery('')}
                  />
                )}
              </div>

              <div className="input-group" style={{ width: 'auto' }}>
                <Filter className="input-icon" size={18} />
                <select 
                  className="premium-input"
                  style={{ padding: '0.75rem 2.5rem 0.75rem 0.5rem', cursor: 'pointer', appearance: 'none' }}
                  value={filterClass}
                  onChange={(e) => setFilterClass(e.target.value)}
                >
                  {uniqueClassifications.map(c => (
                    <option key={c} value={c} style={{ background: '#1a1b1e' }}>{c}</option>
                  ))}
                </select>
              </div>

              <div className="input-group" style={{ width: 'auto' }}>
                <Calendar className="input-icon" size={18} />
                <input 
                  type="date" 
                  className="premium-input"
                  style={{ padding: '0.75rem 1rem 0.75rem 0.5rem', cursor: 'pointer' }}
                  value={filterDate}
                  onChange={(e) => setFilterDate(e.target.value)}
                />
                {filterDate && (
                  <X 
                    size={16} 
                    style={{ marginRight: '1rem', cursor: 'pointer', color: 'var(--text-muted)' }} 
                    onClick={() => setFilterDate('')}
                  />
                )}
              </div>
            </div>
          </div>
          
          <div style={{ overflowY: 'auto', maxHeight: '600px', paddingRight: '0.5rem' }}>
            {filteredHistory.length === 0 ? (
              <div className="text-center mt-2" style={{ padding: '4rem 2rem', background: 'rgba(0,0,0,0.1)', borderRadius: '16px' }}>
                <FileWarning size={48} style={{ marginBottom: '1.5rem', opacity: 0.2 }} />
                <p className="text-muted" style={{ fontSize: '1.1rem' }}>
                  {history.length === 0 ? 'No historical data available' : 'No records match your search or filters'}
                </p>
                {(searchQuery || filterClass !== 'All' || filterDate) && (
                  <button 
                    className="btn btn-secondary mt-1" 
                    onClick={() => { setSearchQuery(''); setFilterClass('All'); setFilterDate(''); }}
                  >
                    Clear All Filters
                  </button>
                )}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {filteredHistory.map((item) => (
                  <div key={item._id} className="result-card" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', transition: 'all 0.2s ease', margin: 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Activity size={20} style={{ color: item.status && item.status.toLowerCase() === 'completed' ? 'var(--primary)' : 'var(--warning)' }} />
                        </div>
                        <div>
                          <strong className="font-important" style={{ fontSize: '1.125rem', color: 'var(--text-main)' }}>{item.cowId}</strong>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.125rem' }}>
                            {new Date(item.createdAt).toLocaleString()}
                          </div>
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
                        {getStatusBadge(item.status)}
                        {item.status && item.status.toLowerCase() === 'completed' && (
                          <button 
                            onClick={() => handleDownloadPDF(item)} 
                            style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '8px', color: 'var(--primary)', cursor: 'pointer', width: '36px', height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s ease' }}
                            title="Download PDF Report"
                            onMouseOver={(e) => { e.currentTarget.style.background = 'var(--primary)'; e.currentTarget.style.color = 'white'; }}
                            onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(16, 185, 129, 0.1)'; e.currentTarget.style.color = 'var(--primary)'; }}
                          >
                            <Download size={18} />
                          </button>
                        )}
                        <button 
                          onClick={() => handleDelete(item._id)} 
                          style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '8px', color: 'var(--error)', cursor: 'pointer', width: '36px', height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s ease' }}
                          title="Remove Result"
                          onMouseOver={(e) => { e.currentTarget.style.background = 'var(--error)'; e.currentTarget.style.color = 'white'; }}
                          onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)'; e.currentTarget.style.color = 'var(--error)'; }}
                        >
                          <Trash2 size={18} />
                        </button>
                      </div>
                    </div>
                    {item.classification && (
                      <div style={{ marginTop: '1.25rem', padding: '1.25rem', background: 'var(--border)', borderRadius: '12px', borderLeft: '4px solid var(--primary)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                        <div>
                          <div className="result-label">Classification</div>
                          <div className="font-important" style={{ fontSize: '1.125rem', color: 'var(--text-main)' }}>{item.classification}</div>
                          {item.confidence !== undefined && (
                            <div className="font-important" style={{ fontSize: '0.875rem', color: 'var(--primary)', marginTop: '0.25rem' }}>{((item.confidence || 0) * 100).toFixed(1)}% Confidence</div>
                          )}
                        </div>
                        
                        {/* ✅ FIXED READ FROM yield_litres */}
                        {item.yield_litres !== undefined && item.yield_litres !== null && (
                          <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: '1.5rem' }}>
                             <div className="result-label">Predicted Yield</div>
                             <div className="font-important" style={{ fontSize: '1.125rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                               <Milk size={18} style={{ color: 'var(--primary)' }} />
                               {Number(item.yield_litres || 0).toFixed(2)} L/day
                             </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* AI Insight Section */}
                    {item.status && item.status.toLowerCase() === 'completed' && getAIInsight(item) && (
                      <div style={{ 
                        marginTop: '1.25rem', 
                        padding: '1rem 1.25rem', 
                        background: 'var(--border)', 
                        borderRadius: '12px', 
                        border: '1px solid var(--border)',
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '0.875rem'
                      }}>
                        <div style={{ 
                          marginTop: '0.125rem',
                          color: getAIInsight(item).color,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          {getAIInsight(item).icon}
                        </div>
                        <div>
                          <p className="font-important" style={{ 
                            fontSize: '0.875rem', 
                            color: 'var(--text-main)', 
                            marginBottom: '0.25rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem'
                          }}>
                            AI Insight
                          </p>
                          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                            {getAIInsight(item).text}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
