// src/extension.ts
import * as vscode from 'vscode';
import * as path from 'path';
import { ModelClient, ScanResult, ThreatInfo } from './modelClient';

export function activate(context: vscode.ExtensionContext) {
    const provider = new DamlDashboardProvider(context);
    const modelClient = new ModelClient();

    // Register the sidebar provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('daml.dashboardView', provider)
    );

    // Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('daml.toggleStatusView', () => provider.toggleExtension()),
        vscode.commands.registerCommand('daml.showMenu', () => provider.showQuickPickMenu()),
        vscode.commands.registerCommand('daml.scanWorkspace', async () => {
            // Check model server health first
            const health = await modelClient.healthCheck();
            if (!health) {
                modelClient.showConnectionError();
                return;
            }
            provider.simulateScan(modelClient);
        }),
        vscode.commands.registerCommand('daml.scanActiveFile', async () => {
            const health = await modelClient.healthCheck();
            if (!health) {
                modelClient.showConnectionError();
                return;
            }
            provider.scanActiveFile(modelClient);
        }),
        vscode.commands.registerCommand('daml.resolveThreats', () => provider.resolveThreats(modelClient))
    );

    // Auto-check model server on startup
    modelClient.healthCheck().then(health => {
        if (health) {
            vscode.window.showInformationMessage(
                `DAML Model connected: v${health.model_version} (Engine ${health.engine_version})`
            );
        }
    });
}

class DamlDashboardProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private statusBarItem: vscode.StatusBarItem;
    private threatCount: number = 0;
    private isEnabled: boolean = true;
    private lastScanResult?: ScanResult;
    private recentDetections: any[] = [];

    constructor(private context: vscode.ExtensionContext) {
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'daml.showMenu';
        this.updateStatusBar();
        this.statusBarItem.show();
    }

    public resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(this.context.extensionPath, 'media'))]
        };

        webviewView.webview.onDidReceiveMessage(async msg => {
            switch (msg.command) {
                case 'resolveThreats':
                    await this.resolveThreatsWithIds(msg.threatIds);
                    break;
                case 'scanWorkspace':
                    vscode.commands.executeCommand('daml.scanWorkspace');
                    break;
                case 'scanActiveFile':
                    vscode.commands.executeCommand('daml.scanActiveFile');
                    break;
                case 'getStats':
                    // Will be handled by model client
                    break;
            }
        });

        this.updateWebview();
    }

    public toggleExtension() {
        this.isEnabled = !this.isEnabled;
        vscode.window.showInformationMessage(`DAML Protection ${this.isEnabled ? 'Enabled' : 'Disabled'}`);
        this.updateStatusBar();
        this.updateWebview();
    }

    public async showQuickPickMenu() {
        const toggleLabel = this.isEnabled ? '$(circle-slash) Disable' : '$(play) Enable';
        const items = [
            { label: '$(graph) Open Dashboard', cmd: 'daml.dashboardView.focus' },
            { label: '$(shield) Scan Workspace', cmd: 'daml.scanWorkspace' },
            { label: '$(file-code) Scan Active File', cmd: 'daml.scanActiveFile' },
            { label: toggleLabel, cmd: 'daml.toggleStatusView' }
        ];

        const selection = await vscode.window.showQuickPick(items, { title: 'DAML Control' });
        if (selection) vscode.commands.executeCommand(selection.cmd);
    }

    public async simulateScan(modelClient: ModelClient) {
        if (!this.isEnabled) {
            vscode.window.showWarningMessage('Enable DAML first.');
            return;
        }

        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            vscode.window.showWarningMessage('No workspace open');
            return;
        }

        const scanPath = workspaceFolders[0].uri.fsPath;

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "DAML AI Scanning...",
            cancellable: true
        }, async (progress, token) => {
            try {
                progress.report({ increment: 10, message: "Connecting to model..." });
                
                // Perform actual scan
                const result = await modelClient.scanPath(scanPath, 'full');
                this.lastScanResult = result;
                
                progress.report({ increment: 90, message: `Found ${result.threats_found} threats` });

                // Update state
                this.threatCount = result.threats_found;
                
                // Add to recent detections
                const newDetections = result.threats.map(t => ({
                    file: path.basename(t.file),
                    fullPath: t.file,
                    type: t.threat_type,
                    confidence: t.confidence,
                    severity: t.severity,
                    time: 'Just now'
                }));
                this.recentDetections = [...newDetections, ...this.recentDetections].slice(0, 10);

                // Notify webview
                if (this._view) {
                    this._view.webview.postMessage({
                        type: 'scanComplete',
                        payload: {
                            threats: this.threatCount,
                            score: result.score,
                            detections: this.recentDetections,
                            chartData: {
                                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                safe: [result.safe_files, ...Array(6).fill(Math.floor(result.safe_files * 0.8))],
                                threats: [result.threats_found, ...Array(6).fill(Math.floor(result.threats_found * 0.5))]
                            }
                        }
                    });
                }

                this.updateStatusBar();
                this.updateWebview();

                if (result.threats_found > 0) {
                    vscode.window.showWarningMessage(
                        `DAML: Found ${result.threats_found} threats in ${result.total_files} files`,
                        'View Details'
                    ).then(s => s === 'View Details' && this._view?.show());
                } else {
                    vscode.window.showInformationMessage(`DAML: All ${result.total_files} files clean`);
                }

            } catch (error) {
                vscode.window.showErrorMessage(`Scan failed: ${error}`);
            }
        });
    }

    public async scanActiveFile(modelClient: ModelClient) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active file');
            return;
        }

        const filePath = editor.document.uri.fsPath;
        
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Scanning ${path.basename(filePath)}...`
        }, async () => {
            try {
                const result = await modelClient.scanSingleFile(filePath);
                
                if (result.is_threat) {
                    vscode.window.showWarningMessage(
                        `⚠️ Threat detected in ${path.basename(filePath)}: ${result.family} (${(result.confidence * 100).toFixed(1)}%)`,
                        'View Details',
                        'Ignore'
                    );
                    this.threatCount++;
                } else {
                    vscode.window.showInformationMessage(
                        `✅ ${path.basename(filePath)} is clean (${(result.confidence * 100).toFixed(1)}% confidence)`
                    );
                }
                
                this.updateStatusBar();
                this.updateWebview();
                
            } catch (error) {
                vscode.window.showErrorMessage(`File scan failed: ${error}`);
            }
        });
    }

    public async resolveThreats(modelClient: ModelClient) {
        if (!this.lastScanResult || this.lastScanResult.threats.length === 0) {
            vscode.window.showInformationMessage('No threats to resolve');
            return;
        }

        const threatIds = this.lastScanResult.threats.map((t, i) => `threat_${i}`);
        
        try {
            await modelClient.resolveThreats(threatIds);
            this.threatCount = 0;
            this.recentDetections = [];
            this.updateStatusBar();
            this.updateWebview();
            vscode.window.showInformationMessage('DAML: All threats resolved');
            
            if (this._view) {
                this._view.webview.postMessage({ type: 'threatsResolved' });
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to resolve: ${error}`);
        }
    }

    private async resolveThreatsWithIds(threatIds: string[]) {
        this.threatCount = Math.max(0, this.threatCount - threatIds.length);
        this.updateStatusBar();
        this.updateWebview();
    }

    private updateStatusBar() {
        this.statusBarItem.text = this.isEnabled 
            ? `$(shield) ${this.threatCount > 0 ? `$(alert) ${this.threatCount}` : '$(check) Clean'}` 
            : `$(circle-slash) DAML Off`;
        
        this.statusBarItem.backgroundColor = (this.isEnabled && this.threatCount > 0) 
            ? new vscode.ThemeColor('statusBarItem.warningBackground') 
            : undefined;
        
        this.statusBarItem.tooltip = this.isEnabled 
            ? `DAML AI: ${this.threatCount} threats detected` 
            : 'DAML Protection Disabled';
    }

    private updateWebview() {
        if (!this._view) return;
        
        if (!this.isEnabled) {
            this._view.webview.html = this.getDisabledHtml();
            return;
        }

        const scriptUri = this._view.webview.asWebviewUri(
            vscode.Uri.file(path.join(this.context.extensionPath, 'media', 'dashboard.js'))
        );
        
        this._view.webview.html = this.getHtml(this._view.webview, scriptUri);
    }

    private getDisabledHtml(): string {
        return `<!DOCTYPE html>
        <html>
        <body style="display:flex;justify-content:center;align-items:center;height:100vh;opacity:0.5;font-family:var(--vscode-font-family);">
            <div style="text-align:center;">
                <h2>⛔ DAML is disabled</h2>
                <p>Enable protection from the status bar menu</p>
            </div>
        </body>
        </html>`;
    }

    private getHtml(webview: vscode.Webview, scriptUri: vscode.Uri): string {
        const threatCount = this.threatCount;
        const score = this.lastScanResult?.score ?? 98.8;
        const detections = this.recentDetections;
        
        // Generate detections HTML
        const detectionsHtml = detections.length > 0 
            ? detections.map(d => `
                <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition" 
                    data-threat-id="${d.fullPath}">
                    <div class="flex items-center gap-2 overflow-hidden">
                        <i data-lucide="file-code" class="w-3 h-3 ${this.getSeverityColor(d.severity)} flex-shrink-0"></i>
                        <span class="truncate opacity-90" title="${d.fullPath}">${d.file}</span>
                        <span class="text-[9px] px-1 rounded bg-[var(--hover-bg)]">${d.type}</span>
                    </div>
                    <div class="flex items-center gap-2 flex-shrink-0 ml-2">
                        <span class="text-[9px] ${this.getSeverityColor(d.severity)}">${(d.confidence * 100).toFixed(0)}%</span>
                        <span class="opacity-50 text-[9px]">${d.time}</span>
                    </div>
                </li>
            `).join('')
            : '<li class="p-4 text-center opacity-50 text-[10px]">No recent detections</li>';

        return `<!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://unpkg.com/lucide@latest"></script>
            <style>
                :root {
                    --bg: var(--vscode-sideBar-background);
                    --fg: var(--vscode-foreground);
                    --border: var(--vscode-widget-border);
                    --accent: var(--vscode-textLink-foreground);
                    --card-bg: var(--vscode-editor-background);
                    --hover-bg: var(--vscode-list-hoverBackground);
                }
                body { 
                    background-color: var(--bg); 
                    color: var(--fg); 
                    font-family: var(--vscode-font-family); 
                    overflow-x: hidden; 
                    padding: 12px;
                }
                .card { 
                    background: var(--card-bg); 
                    border: 1px solid var(--border); 
                    border-radius: 6px; 
                }
                .chart-container { 
                    height: 160px; 
                    width: 100%; 
                    position: relative; 
                }
                ::-webkit-scrollbar { width: 6px; }
                ::-webkit-scrollbar-track { background: transparent; }
                ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
                .severity-critical { color: #ef4444; }
                .severity-high { color: #f97316; }
                .severity-medium { color: #eab308; }
                .severity-low { color: #3b82f6; }
                .pulse {
                    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: .5; }
                }
            </style>
        </head>
        <body class="flex flex-col h-[100vh]">
            <div class="flex flex-col space-y-4 flex-none">
                
                <div class="flex justify-between items-center border-b border-[var(--border)] pb-2">
                    <div class="flex items-center gap-2">
                        <i data-lucide="shield" class="w-4 h-4 text-blue-400 ${threatCount > 0 ? 'pulse' : ''}"></i>
                        <h1 class="text-xs font-bold uppercase tracking-wider opacity-80">DAML AI Status</h1>
                    </div>
                    <button id="clear-btn" class="bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded text-[10px] transition cursor-pointer ${threatCount === 0 ? 'opacity-50' : ''}">
                        Clear Alerts
                    </button>
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div class="card p-3 flex flex-col items-center justify-center text-center">
                        <p class="text-[10px] opacity-60 uppercase mb-1 flex items-center gap-1">
                            <i data-lucide="alert-triangle" class="w-3 h-3 ${threatCount > 0 ? 'text-red-500' : 'text-green-500'}"></i> 
                            Threats
                        </p>
                        <p class="text-2xl font-bold ${threatCount > 0 ? 'text-red-500' : 'text-green-500'}" id="threat-count">
                            ${threatCount}
                        </p>
                    </div>
                    <div class="card p-3 flex flex-col items-center justify-center text-center">
                        <p class="text-[10px] opacity-60 uppercase mb-1 flex items-center gap-1">
                            <i data-lucide="activity" class="w-3 h-3 text-green-500"></i> 
                            AI Score
                        </p>
                        <p class="text-2xl font-bold ${score > 90 ? 'text-green-500' : score > 70 ? 'text-yellow-500' : 'text-red-500'}" id="score-display">
                            ${score}%
                        </p>
                    </div>
                </div>

                <div class="card p-2 chart-container">
                    <canvas id="weeklyChart"></canvas>
                </div>

                <div>
                    <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">AI Actions</h2>
                    <div class="grid grid-cols-2 gap-2">
                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer transition" 
                                onclick="vscode.postMessage({command: 'scanWorkspace'})">
                            <i data-lucide="brain" class="w-4 h-4 text-purple-400"></i>
                            <span>AI Scan Workspace</span>
                        </button>
                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer transition"
                                onclick="vscode.postMessage({command: 'scanActiveFile'})">
                            <i data-lucide="scan" class="w-4 h-4 text-blue-400"></i>
                            <span>Scan Active File</span>
                        </button>
                    </div>
                </div>
            </div>

            <div class="mt-4 flex flex-col flex-grow overflow-hidden">
                <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">AI Detections (${detections.length})</h2>
                <div class="card p-0 overflow-y-auto flex-grow h-32">
                    <ul class="text-[10px] divide-y divide-[var(--border)]" id="detections-list">
                        ${detectionsHtml}
                    </ul>
                </div>
            </div>

            <div class="mt-3 flex justify-between items-center text-[9px] opacity-50 border-t border-[var(--border)] pt-2 flex-none">
                <span>AI Engine v2.1.4</span>
                <span class="flex items-center gap-1">
                    <i data-lucide="cpu" class="w-3 h-3"></i> 
                    Model Active
                </span>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                
                // Listen for updates from extension
                window.addEventListener('message', event => {
                    const message = event.data;
                    switch (message.type) {
                        case 'scanComplete':
                            updateDashboard(message.payload);
                            break;
                        case 'threatsResolved':
                            clearAllThreats();
                            break;
                    }
                });
                
                function updateDashboard(data) {
                    document.getElementById('threat-count').textContent = data.threats;
                    document.getElementById('score-display').textContent = data.score + '%';
                    
                    // Update chart if provided
                    if (data.chartData && window.weeklyChart) {
                        window.weeklyChart.data.datasets[0].data = data.chartData.safe;
                        window.weeklyChart.data.datasets[1].data = data.chartData.threats;
                        window.weeklyChart.update();
                    }
                    
                    // Update detections list
                    const list = document.getElementById('detections-list');
                    if (data.detections && data.detections.length > 0) {
                        list.innerHTML = data.detections.map(d => \`
                            <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer">
                                <div class="flex items-center gap-2 overflow-hidden">
                                    <i data-lucide="file-code" class="w-3 h-3 text-red-400 flex-shrink-0"></i>
                                    <span class="truncate opacity-90">\${d.file}</span>
                                    <span class="text-[9px] px-1 rounded bg-[var(--hover-bg)]">\${d.type}</span>
                                </div>
                                <div class="flex items-center gap-2 flex-shrink-0 ml-2">
                                    <span class="text-[9px] text-red-400">\${(d.confidence * 100).toFixed(0)}%</span>
                                    <span class="opacity-50 text-[9px]">\${d.time}</span>
                                </div>
                            </li>
                        \`).join('');
                        lucide.createIcons();
                    }
                }
                
                function clearAllThreats() {
                    document.getElementById('threat-count').textContent = '0';
                    document.getElementById('score-display').textContent = '100%';
                    document.getElementById('detections-list').innerHTML = 
                        '<li class="p-4 text-center opacity-50 text-[10px]">No recent detections</li>';
                    document.getElementById('clear-btn').classList.add('opacity-50');
                }
                
                lucide.createIcons();
            </script>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }

    private getSeverityColor(severity: string): string {
        const colors: Record<string, string> = {
            'critical': 'severity-critical',
            'high': 'severity-high',
            'medium': 'severity-medium',
            'low': 'severity-low'
        };
        return colors[severity] || 'text-gray-400';
    }
}
