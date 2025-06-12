// MarklygonAI Backtest Results Manager

class BacktestManager {
    constructor() {
        this.data = null;
        this.filteredData = null;
        this.currentFilter = '';
        this.currentSort = 'return_rate';
        this.sortDirection = 'desc';
        
        this.init();
    }

    async init() {
        try {
            await this.loadData();
            this.setupEventListeners();
        } catch (error) {
            console.error('Failed to initialize BacktestManager:', error);
        }
    }

    async loadData() {
        try {
            this.data = await MarklygonAI.loadJSON('./data/backtest_results.json');
            this.filteredData = [...this.data.backtest_results];
            this.updateDisplay();
        } catch (error) {
            console.error('Error loading backtest data:', error);
            MarklygonAI.showAlert('Failed to load backtest results', 'error');
        }
    }

    setupEventListeners() {
        // Filter by ticker
        const tickerFilter = document.getElementById('ticker-filter');
        if (tickerFilter) {
            tickerFilter.addEventListener('change', (e) => {
                this.currentFilter = e.target.value;
                this.applyFilters();
            });
        }

        // Sort controls
        const sortButtons = document.querySelectorAll('.sort-button[data-sort]');
        sortButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const sortField = e.target.getAttribute('data-sort');
                if (sortField) {
                    this.setSortField(sortField);
                }
            });
        });

        // Sort direction toggle
        const sortDirectionBtn = document.getElementById('sort-direction');
        if (sortDirectionBtn) {
            sortDirectionBtn.addEventListener('click', () => {
                this.toggleSortDirection();
            });
        }

        // Listen for page initialization
        document.addEventListener('pageInitialized', (e) => {
            if (e.detail.pageName === 'backtest') {
                this.initialize();
            }
        });
    }

    initialize() {
        if (this.data) {
            this.updateDisplay();
        } else {
            this.loadData();
        }
    }

    applyFilters() {
        this.filteredData = [...this.data.backtest_results];

        // Apply ticker filter
        if (this.currentFilter) {
            this.filteredData = this.filteredData.filter(result => 
                result.ticker === this.currentFilter
            );
        }

        // Apply current sorting
        this.sortData();
        this.updateDisplay();
    }

    setSortField(field) {
        // Update active sort button
        document.querySelectorAll('.sort-button[data-sort]').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeButton = document.querySelector(`[data-sort="${field}"]`);
        if (activeButton) {
            activeButton.classList.add('active');
        }

        this.currentSort = field;
        this.sortData();
        this.updateDisplay();
    }

    toggleSortDirection() {
        this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        
        const sortDirectionBtn = document.getElementById('sort-direction');
        if (sortDirectionBtn) {
            const icon = sortDirectionBtn.querySelector('i');
            const text = sortDirectionBtn.querySelector('span') || sortDirectionBtn.childNodes[1];
            
            if (this.sortDirection === 'asc') {
                icon.textContent = 'arrow_upward';
                if (text) text.textContent = 'Ascending';
            } else {
                icon.textContent = 'arrow_downward';
                if (text) text.textContent = 'Descending';
            }
        }

        this.sortData();
        this.updateDisplay();
    }

    sortData() {
        this.filteredData.sort((a, b) => {
            let aVal = a[this.currentSort];
            let bVal = b[this.currentSort];

            // Handle null/undefined values
            if (aVal === null || aVal === undefined) aVal = 0;
            if (bVal === null || bVal === undefined) bVal = 0;

            // Numeric comparison for most fields
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return this.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
            }

            // String comparison for text fields
            aVal = String(aVal).toLowerCase();
            bVal = String(bVal).toLowerCase();
            
            if (this.sortDirection === 'asc') {
                return aVal.localeCompare(bVal);
            } else {
                return bVal.localeCompare(aVal);
            }
        });
    }

    updateDisplay() {
        this.updateResultsCount();
        this.renderTable();
    }

    updateResultsCount() {
        const countElement = document.getElementById('results-count');
        if (countElement && this.filteredData) {
            const total = this.data.backtest_results.length;
            const filtered = this.filteredData.length;
            
            if (this.currentFilter) {
                countElement.textContent = `Showing ${filtered} of ${total} results (filtered by ${this.currentFilter})`;
            } else {
                countElement.textContent = `Showing all ${total} results`;
            }
        }
    }

    renderTable() {
        const tableContainer = document.getElementById('backtest-results-table');
        if (!tableContainer || !this.filteredData) return;

        const columns = [
            { key: 'model_name', label: 'Model', type: 'text' },
            { key: 'ticker', label: 'Ticker', type: 'text' },
            { key: 'return_rate', label: 'Return Rate', type: 'percentage' },
            { key: 'max_drawdown', label: 'Max Drawdown', type: 'percentage' },
            { key: 'sharpe_ratio', label: 'Sharpe Ratio', type: 'number', format: { decimals: 2 } },
            { key: 'invalid_actions', label: 'Invalid Actions', type: 'number', format: { decimals: 0 } },
            { key: 'total_trades', label: 'Total Trades', type: 'number', format: { decimals: 0 } },
            { key: 'win_rate', label: 'Win Rate', type: 'percentage' },
            { key: 'profit_factor', label: 'Profit Factor', type: 'number', format: { decimals: 2 } }
        ];

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
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.generateTableRows(this.filteredData, columns)}
                    </tbody>
                </table>
            </div>
        `;

        tableContainer.innerHTML = tableHTML;

        // Add sorting functionality to table headers
        this.addTableSorting(tableContainer, columns);
    }

    generateTableRows(data, columns) {
        return data.map(row => `
            <tr class="hover:bg-gray-50 transition-colors duration-150">
                ${columns.map(col => `
                    <td class="px-6 py-4">
                        ${this.formatCellValue(row[col.key], col.type, col.format, row)}
                    </td>
                `).join('')}
                <td class="px-6 py-4">
                    <div class="flex space-x-2">
                        <button class="btn-sm btn-outline" onclick="BacktestManager.viewDetails(${row.result_id})">
                            <i class="material-icons text-sm">visibility</i>
                        </button>
                        <button class="btn-sm btn-outline" onclick="BacktestManager.downloadReport(${row.result_id})">
                            <i class="material-icons text-sm">download</i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    formatCellValue(value, type, format, row) {
        if (value === null || value === undefined) return '-';

        switch (type) {
            case 'percentage':
                const formattedPercent = MarklygonAI.formatPercentage(value); // value is already a decimal
                const colorClass = value >= 0 ? 'text-green-600' : 'text-red-600';
                // Special handling for max_drawdown to add negative sign
                if (row && row.max_drawdown !== undefined && value === row.max_drawdown) {
                    return `<span class="${colorClass} font-medium">-${(value * 100).toFixed(2)}%</span>`;
                }
                return `<span class="${colorClass} font-medium">${formattedPercent}</span>`;
                
            case 'number':
                const decimals = format?.decimals || 2;
                const formattedNumber = MarklygonAI.formatNumber(value, decimals);
                
                // Special formatting for invalid actions
                if (row && row.invalid_actions !== undefined && value === row.invalid_actions) {
                    const badgeClass = value === 0 ? 'status-success' : value <= 3 ? 'status-warning' : 'status-error';
                    return `<span class="status-badge ${badgeClass}">${formattedNumber}</span>`;
                }
                
                return formattedNumber;
                
            case 'text':
                // Special formatting for model names
                if (row && row.model_name !== undefined && value === row.model_name) {
                    const modelColors = {
                        'Eddie': 'bg-blue-100 text-blue-800',
                        'Mark': 'bg-green-100 text-green-800',
                        'SugarMixCoffee': 'bg-purple-100 text-purple-800',
                        'Mint': 'bg-teal-100 text-teal-800',
                        'Jeawan': 'bg-orange-100 text-orange-800',
                        'BNM': 'bg-gray-100 text-gray-800'
                    };
                    const colorClass = modelColors[value] || 'bg-gray-100 text-gray-800';
                    return `<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClass}">${value}</span>`;
                }
                
                // Special formatting for tickers
                if (row && row.ticker !== undefined && value === row.ticker) {
                    return `<span class="font-mono font-semibold text-gray-900">${value}</span>`;
                }
                
                return value;
                
            default:
                return value;
        }
    }

    addTableSorting(container, columns) {
        const headers = container.querySelectorAll('th[data-sort]');
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const sortKey = header.getAttribute('data-sort');
                this.setSortField(sortKey);
            });
        });
    }

    static viewDetails(resultId) {
        const manager = window.BacktestManager;
        if (!manager || !manager.data) return;

        const result = manager.data.backtest_results.find(r => r.result_id === resultId);
        if (!result) return;

        // Create modal content
        const modalContent = `
            <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50" id="backtest-modal">
                <div class="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="text-lg font-semibold text-gray-900">Backtest Details</h3>
                        <button class="text-gray-400 hover:text-gray-600" onclick="document.getElementById('backtest-modal').remove()">
                            <i class="material-icons">close</i>
                        </button>
                    </div>
                    
                    <div class="space-y-4">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Model</label>
                                <p class="mt-1 text-sm text-gray-900">${result.model_name}</p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Ticker</label>
                                <p class="mt-1 text-sm text-gray-900">${result.ticker}</p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Return Rate</label>
                                <p class="mt-1 text-sm font-semibold ${result.return_rate >= 0 ? 'text-green-600' : 'text-red-600'}">
                                    ${MarklygonAI.formatPercentage(result.return_rate)}
                                </p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Max Drawdown</label>
                                <p class="mt-1 text-sm font-semibold text-red-600">
                                    -${(result.max_drawdown * 100).toFixed(2)}%
                                </p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Sharpe Ratio</label>
                                <p class="mt-1 text-sm text-gray-900">${MarklygonAI.formatNumber(result.sharpe_ratio, 2)}</p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Win Rate</label>
                                <p class="mt-1 text-sm text-gray-900">${result.win_rate.toFixed(1)}%</p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Total Trades</label>
                                <p class="mt-1 text-sm text-gray-900">${result.total_trades.toLocaleString()}</p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Invalid Actions</label>
                                <p class="mt-1 text-sm text-gray-900">${result.invalid_actions}</p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Initial Capital</label>
                                <p class="mt-1 text-sm text-gray-900">${MarklygonAI.formatCurrency(result.initial_capital)}</p>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Final Capital</label>
                                <p class="mt-1 text-sm text-gray-900">${MarklygonAI.formatCurrency(result.final_capital)}</p>
                            </div>
                        </div>
                        
                        <div class="border-t pt-4">
                            <label class="block text-sm font-medium text-gray-700">Test Period</label>
                            <p class="mt-1 text-sm text-gray-900">
                                ${MarklygonAI.formatDate(result.start_date, { year: 'numeric', month: 'long', day: 'numeric' })} - 
                                ${MarklygonAI.formatDate(result.end_date, { year: 'numeric', month: 'long', day: 'numeric' })}
                            </p>
                        </div>
                    </div>
                    
                    <div class="flex justify-end space-x-3 mt-6">
                        <button class="btn btn-outline" onclick="document.getElementById('backtest-modal').remove()">
                            Close
                        </button>
                        <button class="btn btn-primary" onclick="BacktestManager.downloadReport(${resultId})">
                            Download Report
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalContent);
    }

    static downloadReport(resultId) {
        const manager = window.BacktestManager;
        if (!manager || !manager.data) return;

        const result = manager.data.backtest_results.find(r => r.result_id === resultId);
        if (!result) return;

        // Simulate download
        MarklygonAI.showAlert(`Downloading report for ${result.model_name} - ${result.ticker}`, 'info');
        
        // In a real implementation, this would trigger an actual file download
        console.log('Downloading backtest report:', result);
    }

    // Filter data for backend API simulation
    async filterByTicker(ticker) {
        // Simulate API call
        await MarklygonAI.simulateAPI('/api/backtest/filter', { ticker }, 500);
        
        this.currentFilter = ticker;
        this.applyFilters();
        
        MarklygonAI.showAlert(`Filtered results by ticker: ${ticker || 'All'}`, 'success', 3000);
    }

    // Sort data for backend API simulation
    async sortBy(field, direction) {
        // Simulate API call
        await MarklygonAI.simulateAPI('/api/backtest/sort', { field, direction }, 500);
        
        this.currentSort = field;
        this.sortDirection = direction;
        this.sortData();
        this.updateDisplay();
        
        MarklygonAI.showAlert(`Sorted by ${field} (${direction})`, 'success', 3000);
    }
}

// Initialize BacktestManager
window.BacktestManager = new BacktestManager(); 