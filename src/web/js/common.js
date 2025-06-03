// MarklygonAI Common Utilities

class MarklygonAI {
    constructor() {
        this.baseURL = window.location.origin;
        this.currentUser = null;
        this.currentPage = 'dashboard';
        this.cache = new Map();
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadUserData();
    }

    setupEventListeners() {
        // Global error handler
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            this.showAlert('An unexpected error occurred', 'error');
        });

        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            this.showAlert('An unexpected error occurred', 'error');
        });
    }

    // Data Management
    async loadJSON(filePath) {
        try {
            const cacheKey = `json_${filePath}`;
            if (this.cache.has(cacheKey)) {
                return this.cache.get(cacheKey);
            }

            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`Failed to load ${filePath}: ${response.status}`);
            }
            
            const data = await response.json();
            this.cache.set(cacheKey, data);
            return data;
        } catch (error) {
            console.error('Error loading JSON:', error);
            throw error;
        }
    }

    async saveJSON(filePath, data) {
        try {
            // In a real implementation, this would make an API call
            console.log('Saving data to:', filePath, data);
            const cacheKey = `json_${filePath}`;
            this.cache.set(cacheKey, data);
            return true;
        } catch (error) {
            console.error('Error saving JSON:', error);
            throw error;
        }
    }

    // User Management
    async loadUserData() {
        try {
            const userData = await this.loadJSON('./data/user.json');
            this.currentUser = userData;
            this.updateUserDisplay();
        } catch (error) {
            console.warn('User data not found, using default');
            this.currentUser = {
                id: 1,
                username: 'demo_user',
                email: 'demo@marklygonai.com',
                created_at: new Date().toISOString()
            };
        }
    }

    updateUserDisplay() {
        const userElements = document.querySelectorAll('[data-user-field]');
        userElements.forEach(element => {
            const field = element.getAttribute('data-user-field');
            if (this.currentUser && this.currentUser[field]) {
                element.textContent = this.currentUser[field];
            }
        });
    }

    // UI Utilities
    showAlert(message, type = 'info', duration = 5000) {
        const alertContainer = this.getOrCreateAlertContainer();
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} fade-in`;
        alert.innerHTML = `
            <div class="flex justify-between items-center">
                <span>${message}</span>
                <button class="ml-4 text-lg font-bold" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        alertContainer.appendChild(alert);

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (alert.parentElement) {
                    alert.remove();
                }
            }, duration);
        }
    }

    getOrCreateAlertContainer() {
        let container = document.getElementById('alert-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'alert-container';
            container.className = 'fixed top-4 right-4 z-50 space-y-2';
            document.body.appendChild(container);
        }
        return container;
    }

    showLoading(element, message = 'Loading...') {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (element) {
            element.innerHTML = `
                <div class="flex items-center justify-center py-8">
                    <div class="spinner mr-3"></div>
                    <span class="text-gray-600">${message}</span>
                </div>
            `;
        }
    }

    hideLoading(element) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (element) {
            element.innerHTML = '';
        }
    }

    // Format Utilities
    formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(amount);
    }

    formatPercentage(value, decimals = 2) {
        return `${(value * 100).toFixed(decimals)}%`;
    }

    formatNumber(number, decimals = 2) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(number);
    }

    formatDate(date, options = {}) {
        const defaultOptions = {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        
        const formatOptions = { ...defaultOptions, ...options };
        return new Intl.DateTimeFormat('en-US', formatOptions).format(new Date(date));
    }

    // Chart Utilities
    getChartColors() {
        return {
            primary: '#2563eb',    // blue-600
            secondary: '#7c3aed',  // purple-600
            success: '#059669',    // green-600
            warning: '#d97706',    // yellow-600
            danger: '#dc2626',     // red-600
            info: '#0891b2',       // cyan-600
            light: '#6b7280',      // gray-500
            dark: '#1f2937'        // gray-800
        };
    }

    createChart(canvasId, config) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        // Destroy existing chart if it exists
        const existingChart = Chart.getChart(canvas);
        if (existingChart) {
            existingChart.destroy();
        }

        // Set default chart options
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    grid: {
                        color: '#f3f4f6'
                    }
                }
            }
        };

        // Merge configurations
        config.options = { ...defaultOptions, ...config.options };

        return new Chart(canvas, config);
    }

    // Table Utilities
    createTable(containerId, data, columns, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container with id '${containerId}' not found`);
            return;
        }

        const tableHTML = `
            <div class="overflow-x-auto">
                <table class="table">
                    <thead>
                        <tr>
                            ${columns.map(col => `
                                <th class="cursor-pointer" data-sort="${col.key}">
                                    ${col.label}
                                    <i class="material-icons text-xs ml-1">unfold_more</i>
                                </th>
                            `).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${this.generateTableRows(data, columns)}
                    </tbody>
                </table>
            </div>
        `;

        container.innerHTML = tableHTML;

        // Add sorting functionality
        this.addTableSorting(container, data, columns);
    }

    generateTableRows(data, columns) {
        return data.map(row => `
            <tr>
                ${columns.map(col => `
                    <td>
                        ${this.formatCellValue(row[col.key], col.type, col.format)}
                    </td>
                `).join('')}
            </tr>
        `).join('');
    }

    formatCellValue(value, type, format) {
        if (value === null || value === undefined) return '-';

        switch (type) {
            case 'currency':
                return this.formatCurrency(value);
            case 'percentage':
                return this.formatPercentage(value / 100);
            case 'number':
                return this.formatNumber(value, format?.decimals || 2);
            case 'date':
                return this.formatDate(value);
            case 'status':
                return `<span class="status-badge status-${value.toLowerCase()}">${value}</span>`;
            default:
                return value;
        }
    }

    addTableSorting(container, data, columns) {
        const headers = container.querySelectorAll('th[data-sort]');
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const sortKey = header.getAttribute('data-sort');
                const column = columns.find(col => col.key === sortKey);
                
                if (column) {
                    this.sortTable(container, data, column);
                }
            });
        });
    }

    sortTable(container, data, column, direction = 'asc') {
        const sortedData = [...data].sort((a, b) => {
            let aVal = a[column.key];
            let bVal = b[column.key];

            // Handle null/undefined values
            if (aVal === null || aVal === undefined) aVal = '';
            if (bVal === null || bVal === undefined) bVal = '';

            // Numeric comparison
            if (column.type === 'number' || column.type === 'currency' || column.type === 'percentage') {
                aVal = parseFloat(aVal) || 0;
                bVal = parseFloat(bVal) || 0;
                return direction === 'asc' ? aVal - bVal : bVal - aVal;
            }

            // Date comparison
            if (column.type === 'date') {
                aVal = new Date(aVal);
                bVal = new Date(bVal);
                return direction === 'asc' ? aVal - bVal : bVal - aVal;
            }

            // String comparison
            aVal = String(aVal).toLowerCase();
            bVal = String(bVal).toLowerCase();
            
            if (direction === 'asc') {
                return aVal.localeCompare(bVal);
            } else {
                return bVal.localeCompare(aVal);
            }
        });

        // Update table body
        const tbody = container.querySelector('tbody');
        tbody.innerHTML = this.generateTableRows(sortedData, columns);
    }

    // Utility Functions
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    generateId() {
        return 'id_' + Math.random().toString(36).substr(2, 9);
    }

    // API Simulation (for demo purposes)
    async simulateAPI(endpoint, data = null, delay = 1000) {
        return new Promise((resolve) => {
            setTimeout(() => {
                console.log(`API call to ${endpoint}`, data);
                resolve({ success: true, data: data });
            }, delay);
        });
    }
}

// Initialize the application
const app = new MarklygonAI();

// Export for use in other modules
window.MarklygonAI = app; 