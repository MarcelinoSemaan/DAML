import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

export function activate(context: vscode.ExtensionContext) {
    const provider = new DamlDashboardProvider(context);

    // Register the sidebar provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('daml.dashboardView', provider)
    );

    // Command to toggle protection from the status bar or menu
    context.subscriptions.push(
        vscode.commands.registerCommand('daml.toggleStatusView', () => provider.toggleExtension()),
        vscode.commands.registerCommand('daml.showMenu', () => provider.showQuickPickMenu()),
        vscode.commands.registerCommand('daml.scanWorkspace', () => provider.scanWorkspace()),
        vscode.commands.registerCommand('daml.scanActiveFile', () => provider.scanActiveFile())
    );
}

interface ThreatResult {
    fileName: string;
    score: number;
    timestamp: string;
}

class DamlDashboardProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private statusBarItem: vscode.StatusBarItem;
    private threatCount: number = 0;
    private isEnabled: boolean = true;
    private recentDetections: ThreatResult[] = [];
    private scanHistory: number[] = [12, 19, 8, 15, 22, 30, 45]; // For chart

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

        webviewView.webview.onDidReceiveMessage(msg => {
            if (msg.command === 'resolveThreats') this.resolveThreats();
            if (msg.command === 'scanWorkspace') this.scanWorkspace();
            if (msg.command === 'scanActiveFile') this.scanActiveFile();
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

    /**
     * Scan the currently active/open file in the editor
     */
    public async scanActiveFile() {
        if (!this.isEnabled) {
            return vscode.window.showWarningMessage('Enable DAML first.');
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return vscode.window.showWarningMessage('No active file to scan. Open a file first.');
        }

        const document = editor.document;
        const fileName = path.basename(document.fileName);
        const content = document.getText();

        await this.performScan([{ uri: document.uri, content, fileName }]);
    }

    /**
     * Scan all relevant files in the workspace
     */
    public async scanWorkspace() {
        if (!this.isEnabled) {
            return vscode.window.showWarningMessage('Enable DAML first.');
        }

        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            return vscode.window.showWarningMessage('No workspace folder open.');
        }

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "DAML: Scanning workspace for threats...",
            cancellable: true
        }, async (progress, token) => {
            try {
                // Define file patterns to scan (code files)
                const includePatterns = [
                    '**/*.{js,ts,jsx,tsx,py,java,c,cpp,h,hpp,go,rs,php,rb,cs}',
                    '!**/node_modules/**',
                    '!**/dist/**',
                    '!**/build/**',
                    '!**/.git/**',
                    '!**/out/**',
                    '!**/.vscode/**'
                ];

                // Find all files
                const files = await vscode.workspace.findFiles(
                    '**/*.{js,ts,jsx,tsx,py,java,c,cpp,h,hpp,go,rs,php,rb,cs}',
                    '{**/node_modules/**,**/dist/**,**/build/**,**/.git/**,**/out/**}'
                );

                if (files.length === 0) {
                    vscode.window.showInformationMessage('No code files found in workspace.');
                    return;
                }

                vscode.window.showInformationMessage(`Found ${files.length} files to analyze...`);

                // Process files in batches to avoid overwhelming the system
                const batchSize = 10;
                const filesToProcess: { uri: vscode.Uri; content: string; fileName: string }[] = [];

                for (let i = 0; i < Math.min(files.length, 50); i++) { // Limit to 50 files for performance
                    if (token.isCancellationRequested) break;

                    const file = files[i];
                    try {
                        const content = await this.readFileContent(file);
                        if (content && content.length > 0) {
                            filesToProcess.push({
                                uri: file,
                                content,
                                fileName: path.basename(file.fsPath)
                            });
                        }
                    } catch (err) {
                        console.error(`Failed to read ${file.fsPath}:`, err);
                    }

                    progress.report({ 
                        increment: (100 / Math.min(files.length, 50)), 
                        message: `Reading ${path.basename(file.fsPath)}...` 
                    });
                }

                if (filesToProcess.length > 0) {
                    await this.performScan(filesToProcess);
                }

            } catch (error: any) {
                vscode.window.showErrorMessage(`Scan Error: ${error.message}`);
            }
        });
    }

    /**
     * Read file content using VS Code API
     */
    private async readFileContent(uri: vscode.Uri): Promise<string> {
        try {
            // Try to open as text document first (handles encoding properly)
            const document = await vscode.workspace.openTextDocument(uri);
            return document.getText();
        } catch {
            // Fallback to file system read for binary or special files
            const bytes = await vscode.workspace.fs.readFile(uri);
            return new TextDecoder().decode(bytes);
        }
    }

    /**
     * Perform the actual AI analysis on file contents
     */
    private async performScan(files: { uri: vscode.Uri; content: string; fileName: string }[]) {
    const threats: ThreatResult[] = [];
    let processedCount = 0;

    for (const file of files) {
        try {
            console.log(`[DAML] Scanning ${file.fileName} - Content length: ${file.content.length}`);
            console.log(`[DAML] First 100 chars: ${file.content.substring(0, 100).replace(/\n/g, '\\n')}`);
            
            // Extract features from code and prepare for LSTM
            const featureVector = this.extractFeatures(file.content);
            
            console.log(`[DAML] Feature vector length: ${featureVector.length}`);
            console.log(`[DAML] First 10 features: ${featureVector.slice(0, 10)}`);
            console.log(`[DAML] Last 10 features: ${featureVector.slice(-10)}`);
            
            // Send to your FastAPI backend
            const response = await fetch('http://127.0.0.1:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ raw_sequence: featureVector })
            });

            if (!response.ok) {
                console.error(`[DAML] Server error for ${file.fileName}: ${response.status}`);
                throw new Error(`Server Error: ${response.status}`);
            }

            const result = await response.json() as { status: string; threat_score: number[][] };
            const score = result.threat_score[0][0];
            
            console.log(`[DAML] ${file.fileName} - Threat score: ${score} (${(score * 100).toFixed(2)}%)`);

            // If threat detected (score > 0.5)
            if (score > 0.5) {
                console.log(`[DAML] 🚨 THREAT DETECTED in ${file.fileName}`);
                threats.push({
                    fileName: file.fileName,
                    score: score,
                    timestamp: new Date().toLocaleTimeString()
                });
                this.threatCount++;
            } else {
                console.log(`[DAML] ✅ No threat in ${file.fileName}`);
            }

            processedCount++;

        } catch (error: any) {
            console.error(`[DAML] Analysis failed for ${file.fileName}:`, error.message);
            vscode.window.showErrorMessage(`Scan failed for ${file.fileName}: ${error.message}`);
        }
    }

    console.log(`[DAML] Scan complete. Processed: ${processedCount}, Threats: ${threats.length}`);

    // Update recent detections
    this.recentDetections = [...threats, ...this.recentDetections].slice(0, 10);
    
    // Update chart data with latest threat count
    this.scanHistory.shift();
    this.scanHistory.push(this.threatCount);

    this.updateStatusBar();
    this.updateWebview();

    if (threats.length > 0) {
        vscode.window.showWarningMessage(`DAML: Detected ${threats.length} potential threats!`);
    } else {
        vscode.window.showInformationMessage(`DAML: Scanned ${processedCount} files. No threats detected.`);
    }
}

    /**
     * Extract numerical features from code content for LSTM input
     * Transforms code into (10, 2568) shape expected by your model
     */
    private extractFeatures(content: string): number[] {
        const features: number[] = [];
        
        // Normalize content length - split into lines and pad/truncate
        const lines = content.split('\n');
        const maxLines = 10; // 10 timesteps
        const featuresPerLine = 2568; // Features per timestep
        
        for (let i = 0; i < maxLines; i++) {
            const line = i < lines.length ? lines[i] : '';
            const lineFeatures = this.extractLineFeatures(line, featuresPerLine);
            features.push(...lineFeatures);
        }

        return features;
    }

    /**
     * Extract features from a single line of code
     */
    private extractLineFeatures(line: string, targetLength: number): number[] {
        const features: number[] = [];
        
        // Character-level encoding (ASCII normalized to 0-1)
        for (let i = 0; i < targetLength; i++) {
            if (i < line.length) {
                // Normalize ASCII value to 0-1 range
                features.push(line.charCodeAt(i) / 255);
            } else {
                // Padding
                features.push(0);
            }
        }

        return features;
    }

    private resolveThreats() {
        this.threatCount = 0;
        this.recentDetections = [];
        this.updateStatusBar();
        this.updateWebview();
        vscode.window.showInformationMessage('DAML: All threats resolved.');
    }

    private updateStatusBar() {
        this.statusBarItem.text = this.isEnabled 
            ? `$(shield) ${this.threatCount} Threats` 
            : `$(circle-slash) DAML Off`;
        this.statusBarItem.backgroundColor = (this.isEnabled && this.threatCount > 0) 
            ? new vscode.ThemeColor('statusBarItem.warningBackground') 
            : undefined;
    }

    private updateWebview(currentScore: number = 0) {
        if (!this._view) return;
        
        if (!this.isEnabled) {
            this._view.webview.html = `<body style="display:flex;justify-content:center;align-items:center;height:100vh;opacity:0.5;background:var(--vscode-sideBar-background);color:var(--vscode-foreground);">DAML is disabled</body>`;
            return;
        }

        const scriptUri = this._view.webview.asWebviewUri(
            vscode.Uri.file(path.join(this.context.extensionPath, 'media', 'dashboard.js'))
        );
        
        this._view.webview.html = this.getHtml(this._view.webview, scriptUri, currentScore);
    }

    private getHtml(webview: vscode.Webview, scriptUri: vscode.Uri, currentScore: number = 0) {
        const detectionsHtml = this.recentDetections.map(d => `
            <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition">
                <div class="flex items-center gap-2 overflow-hidden">
                    <i data-lucide="file-code" class="w-3 h-3 ${d.score > 0.7 ? 'text-red-500' : 'text-orange-400'} flex-shrink-0"></i>
                    <span class="truncate opacity-90">${d.fileName}</span>
                </div>
                <span class="text-red-400 font-bold flex-shrink-0 ml-2">${(d.score * 100).toFixed(0)}%</span>
            </li>
        `).join('');

        const historyData = JSON.stringify(this.scanHistory);

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
            </style>
        </head>
        <body class="flex flex-col h-[100vh]">
            <div class="flex flex-col space-y-4 flex-none">
                
                <div class="flex justify-between items-center border-b border-[var(--border)] pb-2">
                    <div class="flex items-center gap-2">
                        <i data-lucide="shield" class="w-4 h-4 text-blue-400"></i>
                        <h1 class="text-xs font-bold uppercase tracking-wider opacity-80">DAML Status</h1>
                    </div>
                    <button id="clear-btn" class="bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded text-[10px] transition cursor-pointer">
                        Clear Alerts
                    </button>
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div class="card p-3 flex flex-col items-center justify-center text-center">
                        <p class="text-[10px] opacity-60 uppercase mb-1 flex items-center gap-1">
                            <i data-lucide="alert-triangle" class="w-3 h-3 text-red-500"></i> Threats
                        </p>
                        <p class="text-2xl font-bold ${this.threatCount > 0 ? 'text-red-500' : 'text-green-500'}">
                            ${this.threatCount}
                        </p>
                    </div>
                    <div class="card p-3 flex flex-col items-center justify-center text-center">
                        <p class="text-[10px] opacity-60 uppercase mb-1 flex items-center gap-1">
                            <i data-lucide="activity" class="w-3 h-3 text-blue-400"></i> Scanned
                        </p>
                        <p class="text-2xl font-bold text-blue-400">${this.recentDetections.length}</p>
                    </div>
                </div>

                <div class="card p-2 chart-container">
                    <canvas id="weeklyChart"></canvas>
                </div>

                <div>
                    <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">Quick Actions</h2>
                    <div class="grid grid-cols-2 gap-2">
                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer transition" onclick="vscode.postMessage({command: 'scanWorkspace'})">
                            <i data-lucide="folder-search" class="w-4 h-4 text-blue-400"></i>
                            <span>Scan Workspace</span>
                        </button>
                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer transition" onclick="vscode.postMessage({command: 'scanActiveFile'})">
                            <i data-lucide="file-search" class="w-4 h-4 text-purple-400"></i>
                            <span>Scan Active File</span>
                        </button>
                    </div>
                </div>
            </div>

            <div class="mt-4 flex flex-col flex-grow overflow-hidden">
                <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">Recent Detections</h2>
                <div class="card p-0 overflow-y-auto flex-grow h-32">
                    <ul class="text-[10px] divide-y divide-[var(--border)]">
                        ${detectionsHtml || '<li class="p-4 text-center opacity-50">No threats detected yet</li>'}
                    </ul>
                </div>
            </div>

            <div class="mt-3 flex justify-between items-center text-[9px] opacity-50 border-t border-[var(--border)] pt-2 flex-none">
                <span>Engine v2.1.4</span>
                <span class="flex items-center gap-1"><i data-lucide="check-circle" class="w-3 h-3"></i> AI Model Active</span>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                lucide.createIcons();
                
                // Initialize chart
                const ctx = document.getElementById('weeklyChart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        datasets: [{
                            label: 'Threats',
                            data: ${historyData},
                            borderColor: 'rgb(239, 68, 68)',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { beginAtZero: true, grid: { color: 'var(--border)' } },
                            x: { grid: { display: false } }
                        }
                    }
                });

                document.getElementById('clear-btn').addEventListener('click', () => {
                    vscode.postMessage({command: 'resolveThreats'});
                });
            </script>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}