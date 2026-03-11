// src/modelClient.ts
import * as vscode from 'vscode';
import fetch from 'node-fetch';

// Types matching FastAPI models
export interface ThreatInfo {
    file: string;
    threat_type: string;
    confidence: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
    timestamp: string;
}

export interface ScanResult {
    scan_id: string;
    total_files: number;
    threats_found: number;
    safe_files: number;
    score: number;
    threats: ThreatInfo[];
    summary: string;
}

export interface HealthStatus {
    status: string;
    model_loaded: boolean;
    model_version: string;
    engine_version: string;
}

export interface StatsData {
    daily: {
        labels: string[];
        safe: number[];
        threats: number[];
    };
    top_families: { name: string; count: number }[];
}

export class ModelClient {
    private baseUrl: string = 'http://127.0.0.1:8000';
    private outputChannel: vscode.OutputChannel;

    constructor() {
        this.outputChannel = vscode.window.createOutputChannel('DAML Model');
    }

    async healthCheck(): Promise<HealthStatus | null> {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            
            if (!response.ok) return null;
            return await response.json() as HealthStatus;
        } catch (error) {
            this.outputChannel.appendLine(`Health check failed: ${error}`);
            return null;
        }
    }

    async scanPath(path: string, scanType: 'full' | 'quick' | 'file' = 'full'): Promise<ScanResult> {
        const response = await fetch(`${this.baseUrl}/scan`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ path, scan_type: scanType })
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Scan failed: ${error}`);
        }

        return await response.json() as ScanResult;
    }

    async scanSingleFile(filePath: string): Promise<any> {
        const response = await fetch(`${this.baseUrl}/scan/file?file_path=${encodeURIComponent(filePath)}`, {
            method: 'POST',
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`File scan failed: ${await response.text()}`);
        }

        return await response.json();
    }

    async getStats(): Promise<StatsData> {
        const response = await fetch(`${this.baseUrl}/stats`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`Failed to get stats: ${await response.text()}`);
        }

        return await response.json() as StatsData;
    }

    async resolveThreats(threatIds: string[]): Promise<{ resolved: number; status: string }> {
        const response = await fetch(`${this.baseUrl}/resolve`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(threatIds)
        });

        if (!response.ok) {
            throw new Error(`Failed to resolve threats: ${await response.text()}`);
        }

        return await response.json();
    }

    showConnectionError(): void {
        vscode.window.showErrorMessage(
            'DAML Model server not running. Start it with: python src/python/model_server.py',
            'Show Instructions'
        ).then(selection => {
            if (selection === 'Show Instructions') {
                const panel = vscode.window.createWebviewPanel(
                    'damlInstructions',
                    'DAML Setup Instructions',
                    vscode.ViewColumn.One,
                    {}
                );
                panel.webview.html = `
                    <h2>Start the Model Server</h2>
                    <pre>cd src/python
pip install fastapi uvicorn pydantic
python model_server.py</pre>
                    <p>The server will start on http://127.0.0.1:8000</p>
                `;
            }
        });
    }
}
