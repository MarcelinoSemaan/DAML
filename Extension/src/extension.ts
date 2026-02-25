import * as vscode from 'vscode';
import * as path from 'path';

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
        vscode.commands.registerCommand('daml.scanWorkspace', () => provider.simulateScan())
    );
}

class DamlDashboardProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private statusBarItem: vscode.StatusBarItem;
    private threatCount: number = 14;
    private isEnabled: boolean = true;

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
            if (msg.command === 'scanWorkspace') this.simulateScan();
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
            { label: toggleLabel, cmd: 'daml.toggleStatusView' }
        ];

        const selection = await vscode.window.showQuickPick(items, { title: 'DAML Control' });
        if (selection) vscode.commands.executeCommand(selection.cmd);
    }

    public simulateScan() {
        if (!this.isEnabled) return vscode.window.showWarningMessage('Enable DAML first.');
        
        vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "DAML Scanning..." }, async () => {
            await new Promise(r => setTimeout(r, 1500));
            this.threatCount += 2;
            this.updateStatusBar();
            this.updateWebview();
        });
    }

    private resolveThreats() {
        this.threatCount = 0;
        this.updateStatusBar();
        this.updateWebview();
        vscode.window.showInformationMessage('DAML: Threats resolved.');
    }

    private updateStatusBar() {
        this.statusBarItem.text = this.isEnabled 
            ? `$(shield-alerts) ${this.threatCount} Threats` 
            : `$(circle-slash) DAML Off`;
        this.statusBarItem.backgroundColor = (this.isEnabled && this.threatCount > 0) 
            ? new vscode.ThemeColor('statusBarItem.warningBackground') 
            : undefined;
    }

    private updateWebview() {
        if (!this._view) return;
        
        if (!this.isEnabled) {
            this._view.webview.html = `<body style="display:flex;justify-content:center;align-items:center;height:100vh;opacity:0.5;">DAML is disabled</body>`;
            return;
        }

        const scriptUri = this._view.webview.asWebviewUri(
            vscode.Uri.file(path.join(this.context.extensionPath, 'media', 'dashboard.js'))
        );
        this._view.webview.html = this.getHtml(this._view.webview, scriptUri);
    }

    private getHtml(webview: vscode.Webview, scriptUri: vscode.Uri) {
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
                /* Custom scrollbar for the log */
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
                            <i data-lucide="activity" class="w-3 h-3 text-green-500"></i> Score
                        </p>
                        <p class="text-2xl font-bold text-green-500">98.8%</p>
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
                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer transition">
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
                        <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition">
                            <div class="flex items-center gap-2 overflow-hidden">
                                <i data-lucide="file-code" class="w-3 h-3 text-red-400 flex-shrink-0"></i>
                                <span class="truncate opacity-90">auth_controller.ts</span>
                            </div>
                            <span class="opacity-50 flex-shrink-0 ml-2">2m ago</span>
                        </li>
                        <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition">
                            <div class="flex items-center gap-2 overflow-hidden">
                                <i data-lucide="file-json" class="w-3 h-3 text-orange-400 flex-shrink-0"></i>
                                <span class="truncate opacity-90">database_config.json</span>
                            </div>
                            <span class="opacity-50 flex-shrink-0 ml-2">1h ago</span>
                        </li>
                        <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition">
                            <div class="flex items-center gap-2 overflow-hidden">
                                <i data-lucide="file-text" class="w-3 h-3 text-yellow-400 flex-shrink-0"></i>
                                <span class="truncate opacity-90">index.html</span>
                            </div>
                            <span class="opacity-50 flex-shrink-0 ml-2">3h ago</span>
                        </li>
                    </ul>
                </div>
            </div>

            <div class="mt-3 flex justify-between items-center text-[9px] opacity-50 border-t border-[var(--border)] pt-2 flex-none">
                <span>Engine v2.1.4</span>
                <span class="flex items-center gap-1"><i data-lucide="check-circle" class="w-3 h-3"></i> Definitions up to date</span>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                lucide.createIcons();
            </script>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}