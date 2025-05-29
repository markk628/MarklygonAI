// My Page JavaScript
class MyPage {
    constructor() {
        this.userProfile = {
            name: 'AI Trader',
            email: 'aitrader@marklygon.ai',
            joinDate: '2025-01-15',
            tradingLevel: 'Advanced',
            preferredModels: ['eddie-001', 'mark-001'],
            riskTolerance: 'moderate',
            notifications: {
                email: true,
                push: true,
                trading: true,
                reports: true
            }
        };
        this.init();
    }

    init() {
        this.renderPage();
        this.setupEventListeners();
    }

    renderPage() {
        const container = document.getElementById('mypage-page');
        container.innerHTML = `
            <!-- Page Header -->
            <div class="mb-8">
                <h1 class="text-3xl font-bold text-gray-900 mb-2">My Account</h1>
                <p class="text-gray-600">Manage your profile and trading preferences</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <!-- Profile Card -->
                <div class="lg:col-span-1">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="text-center">
                            <div class="mx-auto w-24 h-24 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mb-4">
                                <i class="material-icons text-white text-4xl">person</i>
                            </div>
                            <h2 class="text-xl font-semibold text-gray-900">${this.userProfile.name}</h2>
                            <p class="text-gray-600 mb-2">${this.userProfile.email}</p>
                            <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                                ${this.userProfile.tradingLevel} Trader
                            </span>
                        </div>

                        <div class="mt-6 space-y-4">
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-500">Member since</span>
                                <span class="text-sm font-medium text-gray-900">
                                    ${new Date(this.userProfile.joinDate).toLocaleDateString()}
                                </span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-500">Risk Tolerance</span>
                                <span class="text-sm font-medium text-gray-900 capitalize">
                                    ${this.userProfile.riskTolerance}
                                </span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-500">Active Models</span>
                                <span class="text-sm font-medium text-gray-900">
                                    ${this.userProfile.preferredModels.length}
                                </span>
                            </div>
                        </div>

                        <button class="w-full mt-6 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors">
                            Edit Profile
                        </button>
                    </div>

                    <!-- Quick Stats -->
                    <div class="mt-6 bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Trading Statistics</h3>
                        <div class="space-y-4">
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <i class="material-icons text-green-600 mr-2">trending_up</i>
                                    <span class="text-sm text-gray-600">Total Return</span>
                                </div>
                                <span class="text-sm font-semibold text-green-600">+12.04%</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <i class="material-icons text-blue-600 mr-2">psychology</i>
                                    <span class="text-sm text-gray-600">Active Models</span>
                                </div>
                                <span class="text-sm font-semibold text-gray-900">6</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <i class="material-icons text-purple-600 mr-2">analytics</i>
                                    <span class="text-sm text-gray-600">Total Trades</span>
                                </div>
                                <span class="text-sm font-semibold text-gray-900">1,247</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <i class="material-icons text-yellow-600 mr-2">star</i>
                                    <span class="text-sm text-gray-600">Win Rate</span>
                                </div>
                                <span class="text-sm font-semibold text-gray-900">64.2%</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Settings and Preferences -->
                <div class="lg:col-span-2 space-y-6">
                    <!-- Account Settings -->
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Account Settings</h3>
                        <form class="space-y-4">
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                                    <input type="text" value="${this.userProfile.name}" 
                                           class="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                                    <input type="email" value="${this.userProfile.email}" 
                                           class="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                </div>
                            </div>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Trading Level</label>
                                    <select class="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                        <option value="beginner" ${this.userProfile.tradingLevel === 'Beginner' ? 'selected' : ''}>Beginner</option>
                                        <option value="intermediate" ${this.userProfile.tradingLevel === 'Intermediate' ? 'selected' : ''}>Intermediate</option>
                                        <option value="advanced" ${this.userProfile.tradingLevel === 'Advanced' ? 'selected' : ''}>Advanced</option>
                                        <option value="expert" ${this.userProfile.tradingLevel === 'Expert' ? 'selected' : ''}>Expert</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Risk Tolerance</label>
                                    <select class="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                        <option value="conservative" ${this.userProfile.riskTolerance === 'conservative' ? 'selected' : ''}>Conservative</option>
                                        <option value="moderate" ${this.userProfile.riskTolerance === 'moderate' ? 'selected' : ''}>Moderate</option>
                                        <option value="aggressive" ${this.userProfile.riskTolerance === 'aggressive' ? 'selected' : ''}>Aggressive</option>
                                    </select>
                                </div>
                            </div>
                        </form>
                    </div>

                    <!-- Notification Preferences -->
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Notification Preferences</h3>
                        <div class="space-y-4">
                            <div class="flex items-center justify-between">
                                <div>
                                    <h4 class="text-sm font-medium text-gray-900">Email Notifications</h4>
                                    <p class="text-sm text-gray-500">Receive updates via email</p>
                                </div>
                                <label class="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" ${this.userProfile.notifications.email ? 'checked' : ''} 
                                           class="sr-only peer" data-setting="email">
                                    <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                </label>
                            </div>
                            <div class="flex items-center justify-between">
                                <div>
                                    <h4 class="text-sm font-medium text-gray-900">Push Notifications</h4>
                                    <p class="text-sm text-gray-500">Receive browser notifications</p>
                                </div>
                                <label class="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" ${this.userProfile.notifications.push ? 'checked' : ''} 
                                           class="sr-only peer" data-setting="push">
                                    <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                </label>
                            </div>
                            <div class="flex items-center justify-between">
                                <div>
                                    <h4 class="text-sm font-medium text-gray-900">Trading Alerts</h4>
                                    <p class="text-sm text-gray-500">Get notified of important trading events</p>
                                </div>
                                <label class="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" ${this.userProfile.notifications.trading ? 'checked' : ''} 
                                           class="sr-only peer" data-setting="trading">
                                    <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                </label>
                            </div>
                            <div class="flex items-center justify-between">
                                <div>
                                    <h4 class="text-sm font-medium text-gray-900">Weekly Reports</h4>
                                    <p class="text-sm text-gray-500">Receive weekly performance reports</p>
                                </div>
                                <label class="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" ${this.userProfile.notifications.reports ? 'checked' : ''} 
                                           class="sr-only peer" data-setting="reports">
                                    <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                </label>
                            </div>
                        </div>
                    </div>

                    <!-- Preferred Models -->
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Preferred Trading Models</h3>
                        <p class="text-sm text-gray-600 mb-4">Select the AI models you want to use for trading</p>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            ${this.renderModelPreferences()}
                        </div>
                    </div>

                    <!-- Security Settings -->
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Security Settings</h3>
                        <div class="space-y-4">
                            <div>
                                <button class="w-full md:w-auto bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors">
                                    Change Password
                                </button>
                            </div>
                            <div>
                                <button class="w-full md:w-auto bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors">
                                    Enable Two-Factor Authentication
                                </button>
                            </div>
                            <div class="pt-4 border-t border-gray-200">
                                <h4 class="text-sm font-medium text-gray-900 mb-2">Active Sessions</h4>
                                <div class="bg-gray-50 rounded-lg p-3">
                                    <div class="flex items-center justify-between">
                                        <div class="flex items-center">
                                            <i class="material-icons text-green-600 mr-2">computer</i>
                                            <div>
                                                <p class="text-sm font-medium text-gray-900">Current Session</p>
                                                <p class="text-xs text-gray-500">Chrome on Windows • ${new Date().toLocaleString()}</p>
                                            </div>
                                        </div>
                                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                            Active
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Save Button -->
                    <div class="flex justify-end space-x-4">
                        <button class="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors">
                            Cancel
                        </button>
                        <button id="save-settings" class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
                            Save Changes
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    renderModelPreferences() {
        const models = [
            { id: 'eddie-001', name: 'Eddie', description: 'Main signal generator' },
            { id: 'mark-001', name: 'Mark', description: 'Auxiliary model' },
            { id: 'sugarmixcoffee-001', name: 'SugarMixCoffee', description: 'Volatility analysis' },
            { id: 'mint-001', name: 'Mint', description: 'Trend analysis' },
            { id: 'jeawan-001', name: 'Jeawan', description: 'Pattern recognition' },
            { id: 'bnm-001', name: 'BNM', description: 'Market regime detection' }
        ];

        return models.map(model => `
            <div class="border border-gray-200 rounded-lg p-4">
                <div class="flex items-start justify-between">
                    <div class="flex-1">
                        <div class="flex items-center">
                            <input type="checkbox" 
                                   id="model-${model.id}" 
                                   ${this.userProfile.preferredModels.includes(model.id) ? 'checked' : ''}
                                   class="mr-3 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                   data-model="${model.id}">
                            <label for="model-${model.id}" class="text-sm font-medium text-gray-900">
                                ${model.name}
                            </label>
                        </div>
                        <p class="text-xs text-gray-500 mt-1 ml-6">${model.description}</p>
                    </div>
                    <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center ml-2">
                        <i class="material-icons text-blue-600 text-sm">psychology</i>
                    </div>
                </div>
            </div>
        `).join('');
    }

    setupEventListeners() {
        // Save button
        document.getElementById('save-settings').addEventListener('click', () => {
            this.saveSettings();
        });

        // Notification toggles
        document.querySelectorAll('input[data-setting]').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const setting = e.target.dataset.setting;
                this.userProfile.notifications[setting] = e.target.checked;
            });
        });

        // Model preferences
        document.querySelectorAll('input[data-model]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const modelId = e.target.dataset.model;
                if (e.target.checked) {
                    if (!this.userProfile.preferredModels.includes(modelId)) {
                        this.userProfile.preferredModels.push(modelId);
                    }
                } else {
                    const index = this.userProfile.preferredModels.indexOf(modelId);
                    if (index > -1) {
                        this.userProfile.preferredModels.splice(index, 1);
                    }
                }
            });
        });
    }

    saveSettings() {
        // Simulate saving settings
        const saveButton = document.getElementById('save-settings');
        const originalText = saveButton.textContent;
        
        saveButton.textContent = 'Saving...';
        saveButton.disabled = true;

        setTimeout(() => {
            saveButton.textContent = 'Saved!';
            saveButton.className = 'px-6 py-2 bg-green-600 text-white rounded-md';
            
            setTimeout(() => {
                saveButton.textContent = originalText;
                saveButton.className = 'px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors';
                saveButton.disabled = false;
            }, 2000);
        }, 1000);

        // Here you would typically send the data to your backend
        console.log('Settings saved:', this.userProfile);
    }
}

// Initialize when page loads
let myPage;
document.addEventListener('DOMContentLoaded', () => {
    myPage = new MyPage();
}); 