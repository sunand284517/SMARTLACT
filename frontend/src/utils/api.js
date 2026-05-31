// src/api.js

import axios from 'axios';

const api = axios.create({
  baseURL: 'https://smartlact-51.onrender.com/api', // ✅ your backend URL
});

// ✅ Request interceptor (adds token)
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ✅ Response interceptor (handles 401)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login'; // optional redirect
    }
    return Promise.reject(error);
  }
);

export default api;
