import socket
import logging
from orchest_dashboard.app import app, init_host_node

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_lan_ip():
    try:
        dummy = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dummy.connect(("8.8.8.8", 80))
        ip = dummy.getsockname()[0]
        dummy.close()
        return ip
    except Exception:
        # Fallback to local hostname
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

if __name__ == "__main__":
    ip = get_lan_ip()
    node_port = 50006
    dashboard_port = 50000
    
    print("=" * 60)
    print("                 ORCHEST COMPUTING HOST ENGINE")
    print("=" * 60)
    print(f"[*] Determining LAN IP: {ip}")
    print(f"[*] Initializing Host Node Server on port {node_port}...")
    
    init_host_node(ip, node_port, dashboard_port)
    
    print(f"[*] Dashboard UI available at: http://127.0.0.1:{dashboard_port}")
    print(f"[*] Cluster LAN Address: http://{ip}:{dashboard_port}")
    print("=" * 60)
    
    # Start flask app
    app.run(host='0.0.0.0', port=dashboard_port, debug=False, use_reloader=False)
