// MarklygonAI Navigation System

class NavigationManager {
    constructor() {
        this.currentPage = 'dashboard';
        this.pages = new Map();
        this.loadingPages = new Set();
        
        this.init();
    }

    init() {
        this.setupNavigationEvents();
        this.loadInitialPage();
    }

    setupNavigationEvents() {
        // Handle navigation clicks
        document.addEventListener('click', (event) => {
            const navLink = event.target.closest('.nav-link');
            if (navLink) {
                event.preventDefault();
                const targetPage = navLink.getAttribute('data-page');
                if (targetPage) {
                    this.navigateTo(targetPage);
                }
            }
        });

        // Handle browser back/forward buttons
        window.addEventListener('popstate', (event) => {
            const state = event.state;
            if (state && state.page) {
                this.navigateTo(state.page, false);
            }
        });

        // Set initial state
        const initialState = { page: this.currentPage };
        history.replaceState(initialState, '', `#${this.currentPage}`);
    }

    async navigateTo(pageName, updateHistory = true) {
        if (this.currentPage === pageName) return;

        try {
            // Update navigation UI
            this.updateActiveNavigation(pageName);
            
            // Show loading if page needs to be loaded
            if (!this.pages.has(pageName) && !this.loadingPages.has(pageName)) {
                this.showPageLoading(pageName);
                await this.loadPage(pageName);
            }

            // Hide current page
            this.hidePage(this.currentPage);
            
            // Show target page
            this.showPage(pageName);
            
            // Update current page
            this.currentPage = pageName;
            
            // Update browser history
            if (updateHistory) {
                const state = { page: pageName };
                history.pushState(state, '', `#${pageName}`);
            }

            // Update page title
            this.updatePageTitle(pageName);

            // Trigger page-specific initialization
            this.initializePage(pageName);

        } catch (error) {
            console.error('Navigation error:', error);
            MarklygonAI.showAlert(`Failed to load ${pageName} page`, 'error');
        }
    }

    updateActiveNavigation(pageName) {
        // Remove active class from all nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        // Add active class to current nav link
        const activeLink = document.querySelector(`[data-page="${pageName}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }

    showPageLoading(pageName) {
        const pageContainer = document.getElementById(`${pageName}-page`);
        if (pageContainer) {
            MarklygonAI.showLoading(pageContainer, `Loading ${this.getPageTitle(pageName)}...`);
        }
    }

    async loadPage(pageName) {
        if (this.loadingPages.has(pageName)) {
            return; // Already loading
        }

        this.loadingPages.add(pageName);

        try {
            let pageContent = '';

            switch (pageName) {
                case 'portfolio':
                    pageContent = await this.loadPortfolioPage();
                    break;
                case 'backtest':
                    pageContent = await this.loadBacktestPage();
                    break;
                case 'models':
                    pageContent = await this.loadModelsPage();
                    break;
                case 'training':
                    pageContent = await this.loadTrainingPage();
                    break;
                case 'mypage':
                    pageContent = await this.loadMyPage();
                    break;
                default:
                    pageContent = `<div class="text-center py-8">Page "${pageName}" not found</div>`;
            }

            const pageContainer = document.getElementById(`${pageName}-page`);
            if (pageContainer) {
                pageContainer.innerHTML = pageContent;
                this.pages.set(pageName, pageContent);
            }

        } catch (error) {
            console.error(`Error loading ${pageName} page:`, error);
            throw error;
        } finally {
            this.loadingPages.delete(pageName);
        }
    }

    async loadPortfolioPage() {
        return `
            <div class="fade-in">
                <!-- Portfolio Header -->
                <div class="flex justify-between items-center mb-8">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">Portfolio Overview</h1>
                        <p class="text-gray-600 mt-2">Monitor your AI trading models' performance</p>
                    </div>
                    <div class="flex space-x-4">
                        <select id="portfolio-model-filter" class="form-select">
                            <option value="all">All Models</option>
                            <option value="eddie">Eddie</option>
                            <option value="mark">Mark</option>
                            <option value="sugarmixcoffee">SugarMixCoffee</option>
                            <option value="mint">Mint</option>
                            <option value="jeawan">Jeawan</option>
                            <option value="bnm">BNM</option>
                        </select>
                        <button class="btn btn-primary">
                            <i class="material-icons text-sm mr-2">refresh</i>
                            Refresh
                        </button>
                    </div>
                </div>

                <!-- Summary Cards -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div class="metric-card positive">
                        <div class="metric-value positive" id="total-value">$125,430.50</div>
                        <div class="metric-label">Total Portfolio Value</div>
                        <div class="metric-change positive">
                            <i class="material-icons text-xs">trending_up</i>
                            <span>+2.34% ($2,850.20)</span>
                        </div>
                    </div>
                    <div class="metric-card positive">
                        <div class="metric-value positive" id="total-return">+15.67%</div>
                        <div class="metric-label">Total Return</div>
                        <div class="metric-change positive">
                            <i class="material-icons text-xs">trending_up</i>
                            <span>+0.45% today</span>
                        </div>
                    </div>
                    <div class="metric-card neutral">
                        <div class="metric-value neutral" id="active-positions">12</div>
                        <div class="metric-label">Active Positions</div>
                        <div class="metric-change positive">
                            <i class="material-icons text-xs">add</i>
                            <span>2 new today</span>
                        </div>
                    </div>
                    <div class="metric-card neutral">
                        <div class="metric-value neutral" id="win-rate">68.4%</div>
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-change positive">
                            <i class="material-icons text-xs">trending_up</i>
                            <span>+1.2% this week</span>
                        </div>
                    </div>
                </div>

                <!-- Performance Chart -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold text-gray-900">Portfolio Performance</h2>
                        <div class="flex space-x-2">
                            <button class="btn-sm border border-gray-300 rounded-md px-3 py-1 text-sm hover:bg-gray-50">1D</button>
                            <button class="btn-sm border border-gray-300 rounded-md px-3 py-1 text-sm hover:bg-gray-50">1W</button>
                            <button class="btn-sm border border-blue-300 rounded-md px-3 py-1 text-sm bg-blue-50 text-blue-600">1M</button>
                            <button class="btn-sm border border-gray-300 rounded-md px-3 py-1 text-sm hover:bg-gray-50">3M</button>
                            <button class="btn-sm border border-gray-300 rounded-md px-3 py-1 text-sm hover:bg-gray-50">1Y</button>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="portfolio-performance-chart"></canvas>
                    </div>
                </div>

                <!-- Model Performance Grid -->
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold text-gray-900 mb-6">Model Performance</h2>
                    <div id="portfolio-models-grid" class="portfolio-grid">
                        <!-- Models will be loaded here -->
                    </div>
                </div>
            </div>
        `;
    }

    async loadBacktestPage() {
        return `
            <div class="fade-in">
                <!-- Backtest Header -->
                <div class="flex justify-between items-center mb-8">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">Backtest Results</h1>
                        <p class="text-gray-600 mt-2">Analyze historical performance of AI trading models</p>
                    </div>
                    <button class="btn btn-primary">
                        <i class="material-icons text-sm mr-2">analytics</i>
                        Run New Backtest
                    </button>
                </div>

                <!-- Filter Controls -->
                <div class="filter-controls">
                    <div class="filter-group">
                        <label class="filter-label">Filter by Ticker:</label>
                        <select id="ticker-filter" class="form-select w-40">
                            <option value="">All Tickers</option>
                            <option value="NVDA">NVDA</option>
                            <option value="AAPL">AAPL</option>
                            <option value="MSFT">MSFT</option>
                            <option value="AMZN">AMZN</option>
                            <option value="JPM">JPM</option>
                        </select>
                    </div>
                    
                    <div class="filter-group">
                        <label class="filter-label">Sort by:</label>
                        <div class="sort-controls">
                            <button class="sort-button active" data-sort="return_rate">Return Rate</button>
                            <button class="sort-button" data-sort="max_drawdown">Max Drawdown</button>
                            <button class="sort-button" data-sort="sharpe_ratio">Sharpe Ratio</button>
                            <button class="sort-button" data-sort="invalid_actions">Invalid Actions</button>
                        </div>
                    </div>
                    
                    <div class="filter-group">
                        <button id="sort-direction" class="sort-button" data-direction="desc">
                            <i class="material-icons text-sm">arrow_downward</i>
                            Descending
                        </button>
                    </div>
                </div>

                <!-- Results Table -->
                <div class="bg-white rounded-lg shadow-md p-6">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold text-gray-900">Backtest Results</h2>
                        <div class="text-sm text-gray-500">
                            <span id="results-count">Loading results...</span>
                        </div>
                    </div>
                    <div id="backtest-results-table">
                        <!-- Table will be loaded here -->
                    </div>
                </div>
            </div>
        `;
    }

    async loadModelsPage() {
        return `
            <div class="fade-in">
                <div class="flex justify-between items-center mb-8">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">AI Models</h1>
                        <p class="text-gray-600 mt-2">Manage and monitor your trading models</p>
                    </div>
                    <button class="btn btn-primary">
                        <i class="material-icons text-sm mr-2">add</i>
                        Create New Model
                    </button>
                </div>

                <div id="models-grid" class="portfolio-grid">
                    <!-- Models will be loaded here -->
                </div>
            </div>
        `;
    }

    async loadTrainingPage() {
        return `
            <div class="fade-in">
                <div class="flex justify-between items-center mb-8">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">Training Monitor</h1>
                        <p class="text-gray-600 mt-2">Real-time training progress and system status</p>
                    </div>
                    <button class="btn btn-primary">
                        <i class="material-icons text-sm mr-2">play_arrow</i>
                        Start Training
                    </button>
                </div>

                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold text-gray-900 mb-4">Current Training Status</h2>
                    <div class="text-center py-8">
                        <p class="text-gray-600">Training monitor will be implemented here</p>
                    </div>
                </div>
            </div>
        `;
    }

    async loadMyPage() {
        return `
            <div class="fade-in">
                <div class="flex justify-between items-center mb-8">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">My Profile</h1>
                        <p class="text-gray-600 mt-2">Manage your account settings and preferences</p>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold text-gray-900 mb-4">Account Information</h2>
                    <div class="text-center py-8">
                        <p class="text-gray-600">User profile settings will be implemented here</p>
                    </div>
                </div>
            </div>
        `;
    }

    hidePage(pageName) {
        const pageElement = document.getElementById(`${pageName}-page`);
        if (pageElement) {
            pageElement.classList.remove('active');
            pageElement.classList.add('hidden');
        }
    }

    showPage(pageName) {
        const pageElement = document.getElementById(`${pageName}-page`);
        if (pageElement) {
            pageElement.classList.remove('hidden');
            pageElement.classList.add('active');
        }
    }

    updatePageTitle(pageName) {
        const titles = {
            dashboard: 'Dashboard',
            portfolio: 'Portfolio',
            backtest: 'Backtest Results',
            models: 'AI Models',
            training: 'Training Monitor',
            mypage: 'My Profile'
        };
        
        const pageTitle = titles[pageName] || 'MarklygonAI';
        document.title = `${pageTitle} - MarklygonAI`;
    }

    getPageTitle(pageName) {
        const titles = {
            dashboard: 'Dashboard',
            portfolio: 'Portfolio',
            backtest: 'Backtest Results',
            models: 'AI Models',
            training: 'Training Monitor',
            mypage: 'My Profile'
        };
        
        return titles[pageName] || pageName;
    }

    initializePage(pageName) {
        // Trigger page-specific initialization
        const event = new CustomEvent('pageInitialized', {
            detail: { pageName }
        });
        document.dispatchEvent(event);

        // Initialize specific page functionality
        switch (pageName) {
            case 'portfolio':
                this.initializePortfolioPage();
                break;
            case 'backtest':
                this.initializeBacktestPage();
                break;
            case 'models':
                this.initializeModelsPage();
                break;
            // Add other page initializations as needed
        }
    }

    initializePortfolioPage() {
        // This will be implemented in portfolio.js
        if (window.PortfolioManager) {
            window.PortfolioManager.initialize();
        }
    }

    initializeBacktestPage() {
        // This will be implemented in backtest.js
        if (window.BacktestManager) {
            window.BacktestManager.initialize();
        }
    }

    initializeModelsPage() {
        // This will be implemented in models.js
        if (window.ModelsManager) {
            window.ModelsManager.initialize();
        }
    }

    loadInitialPage() {
        // Check URL hash for initial page
        const hash = window.location.hash.substring(1);
        if (hash && hash !== this.currentPage) {
            this.navigateTo(hash, false);
        }
    }
}

// Initialize navigation manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.NavigationManager = new NavigationManager();
}); 