import * as vscode from 'vscode';
import * as path from 'path';

export function activate(context: vscode.ExtensionContext) {
    const daml = new DamlExtension(context);
    
    context.subscriptions.push(
        vscode.commands.registerCommand('daml.showMenu', () => daml.showQuickPickMenu()),
        vscode.commands.registerCommand('daml.showDashboard', () => daml.openDashboard()),
        vscode.commands.registerCommand('daml.scanWorkspace', () => daml.simulateScan()),
        
        vscode.commands.registerCommand('daml.toggleStatusView', () => daml.toggleExtension())
    );
}

class DamlExtension {
    private statusBarItem: vscode.StatusBarItem;
    private threatCount: number = 14; 
    private panel: vscode.WebviewPanel | undefined;
    private isEnabled: boolean = true;

    constructor(private context: vscode.ExtensionContext) {
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'daml.showMenu'; 
        this.updateStatusBar();
        this.statusBarItem.show();
        this.context.subscriptions.push(this.statusBarItem);
    }

    public toggleExtension() {
        this.isEnabled = !this.isEnabled;
        
        if (!this.isEnabled) {
            if (this.panel) this.panel.dispose();
            vscode.window.showInformationMessage('DAML Protection Disabled');
        } else {
            vscode.window.showInformationMessage('DAML Protection Enabled');
        }
        
        this.updateStatusBar();
    }

    public async showQuickPickMenu() {
       
        const toggleLabel = this.isEnabled ? '$(circle-slash) Turn Off Protection' : '$(play) Turn On Protection';

        const items: vscode.QuickPickItem[] = [
            { 
                label: '$(graph) View Dashboard', 
                description: 'Open the protection report',
                detail: 'View detailed analytics and threat history' 
            },
            { 
                label: '$(shield) Scan Workspace', 
                description: 'Run logic analysis' 
            },
            { 
                label: '$(checkall) Resolve All Threats', 
                description: 'Clear current alerts' 
            },
            { label: '', kind: vscode.QuickPickItemKind.Separator },
            { 
                label: toggleLabel,
                description: 'Toggle DAML active state' 
            },
            { label: '$(settings-gear) Settings' }
        ];

        const selection = await vscode.window.showQuickPick(items, {
            title: 'DAML Protection'
        });

        if (!selection) return;

        switch (selection.label) {
            case '$(graph) View Dashboard': this.openDashboard(); break;
            case '$(shield) Scan Workspace': this.simulateScan(); break;
            case '$(checkall) Resolve All Threats': this.resolveThreats(); break;
            case toggleLabel: this.toggleExtension(); break;
            case '$(settings-gear) Settings': 
                vscode.commands.executeCommand('workbench.action.openSettings', 'daml'); 
                break;
        }
    }

    public openDashboard() {
        if (!this.isEnabled) {
            vscode.window.showWarningMessage('DAML is currently disabled.');
            return;
        }

        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'damlDashboard', 'DAML Protection', vscode.ViewColumn.One,
            { 
                enableScripts: true, 
                localResourceRoots: [vscode.Uri.file(path.join(this.context.extensionPath, 'media'))] 
            }
        );

        this.panel.onDidDispose(() => this.panel = undefined);
        this.panel.webview.onDidReceiveMessage(msg => {
            if (msg.command === 'resolveThreats') this.resolveThreats();
        });

        this.updateWebview();
    }

    private resolveThreats() {
        if (!this.isEnabled) return;
        this.threatCount = 0;
        this.updateStatusBar();
        this.updateWebview();
        vscode.window.showInformationMessage('DAML: All threats resolved.');
    }

    public simulateScan() {
        if (!this.isEnabled) {
            vscode.window.showWarningMessage('Enable DAML to perform workspace scans.');
            return;
        }

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "DAML: Analyzing code snippets...",
            cancellable: false
        }, async (progress) => {
            await new Promise(r => setTimeout(r, 1500));
            this.threatCount += 2;
            this.updateStatusBar();
            this.updateWebview();
        });
    }

    private updateStatusBar() {
        if (!this.isEnabled) {
            this.statusBarItem.text = `$(circle-slash) DAML Off`;
            this.statusBarItem.backgroundColor = undefined;
            this.statusBarItem.tooltip = `DAML: Protection is currently inactive.`;
            return;
        }

        if (this.threatCount > 0) {
            this.statusBarItem.text = `$(shield-alerts) ${this.threatCount} Threats`;
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            this.statusBarItem.tooltip = `DAML: ${this.threatCount} security risks found!`;
        } else {
            this.statusBarItem.text = `$(shield-check) DAML Secure`;
            this.statusBarItem.backgroundColor = undefined;
            this.statusBarItem.tooltip = `DAML: Protection Active`;
        }
    }

    private updateWebview() {
        if (this.panel && this.isEnabled) {
            const scriptUri = this.panel.webview.asWebviewUri(
                vscode.Uri.file(path.join(this.context.extensionPath, 'media', 'dashboard.js'))
            );
            this.panel.webview.html = this.getHtml(this.panel.webview, scriptUri);
        }
    }

    private getHtml(webview: vscode.Webview, scriptUri: vscode.Uri) {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                :root {
                    --bg: var(--vscode-editor-background);
                    --fg: var(--vscode-foreground);
                    --card-bg: var(--vscode-sideBar-background);
                    --border: var(--vscode-widget-border);
                    --accent: var(--vscode-textLink-foreground);
                    --btn-bg: var(--vscode-button-background);
                    --btn-fg: var(--vscode-button-foreground);
                }
                body { background-color: var(--bg); color: var(--fg); font-family: var(--vscode-font-family); }
                .dashboard-card { background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; }
                .btn-primary { background-color: var(--btn-bg); color: var(--btn-fg); }
            </style>
        </head>
        <body class="p-8">
            <div class="max-w-4xl mx-auto space-y-6">
                <div class="flex justify-between items-center border-b border-[var(--border)] pb-4">
                    <div>
                        <h1 class="text-xl font-bold">DAML Protection Report</h1>
                        <p class="text-xs opacity-60">Engine v2.1 • Weekly Analysis</p>
                    </div>
                    <button id="clear-btn" class="btn-primary px-4 py-2 rounded text-sm font-medium hover:opacity-90">
                        Mark All Resolved
                    </button>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="dashboard-card p-4">
                        <p class="text-xs opacity-70 uppercase">Total Scanned</p>
                        <p class="text-2xl font-bold mt-1">1,248</p>
                    </div>
                    <div class="dashboard-card p-4 border-l-4 border-l-red-500">
                        <p class="text-xs opacity-70 uppercase">Active Threats</p>
                        <p id="threat-count" class="text-2xl font-bold mt-1 text-red-500">${this.threatCount}</p>
                    </div>
                    <div class="dashboard-card p-4">
                        <p class="text-xs opacity-70 uppercase">Safety Score</p>
                        <p class="text-2xl font-bold mt-1 text-green-500">98.8%</p>
                    </div>
                </div>

                <div class="dashboard-card p-6 h-80">
                    <canvas id="weeklyChart"></canvas>
                </div>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}