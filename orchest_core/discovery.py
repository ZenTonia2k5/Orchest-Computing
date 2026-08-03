import socket
import json
import threading
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("orchest.discovery")

DISCOVERY_PORT = 50005

class OrchestDiscoveryWorker(threading.Thread):
    def __init__(self, node_id, tcp_port, get_metrics_fn=None, on_probe_fn=None):
        super().__init__(daemon=True)
        self.node_id = node_id
        self.tcp_port = tcp_port
        self.get_metrics_fn = get_metrics_fn
        self.on_probe_fn = on_probe_fn
        self.running = False
        self.sock = None

    def run(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass # SO_REUSEPORT not supported on some Windows systems
        
        try:
            self.sock.bind(('', DISCOVERY_PORT))
            logger.info(f"Discovery Worker listening on UDP port {DISCOVERY_PORT}")
        except Exception as e:
            logger.error(f"Failed to bind discovery socket: {e}")
            self.running = False
            return

        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                if not data:
                    continue
                
                try:
                    message = json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    continue

                if message.get("type") == "DISCOVER_PROBE":
                    host_ip = message.get("host_ip")
                    host_port = message.get("host_port")
                    
                    if not host_ip or not host_port:
                        continue
                    
                    if self.on_probe_fn:
                        try:
                            self.on_probe_fn(host_ip)
                        except Exception as ex:
                            logger.error(f"Error executing on_probe_fn: {ex}")
                    
                    # Gather node details
                    metrics = {}
                    if self.get_metrics_fn:
                        try:
                            metrics = self.get_metrics_fn()
                        except Exception as ex:
                            logger.error(f"Error gathering metrics: {ex}")
                    
                    response = {
                        "type": "DISCOVER_RESPONSE",
                        "node_id": self.node_id,
                        "ip": socket.gethostbyname(socket.gethostname()),
                        "port": self.tcp_port,
                        "hostname": socket.gethostname(),
                        "metrics": metrics
                    }
                    
                    # Unicast response back to host
                    resp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    resp_sock.sendto(json.dumps(response).encode('utf-8'), (host_ip, host_port))
                    resp_sock.close()
                    logger.info(f"Responded to discovery probe from {host_ip}:{host_port}")
            except Exception as e:
                if self.running:
                    logger.error(f"Error in discovery worker loop: {e}")
                time.sleep(0.5)

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

class OrchestDiscoveryHost:
    def __init__(self):
        self.discovered_nodes = {}

    def discover(self, timeout=2.0):
        # Create a socket to receive responses
        rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx_sock.bind(('', 0)) # bind to dynamic port
        rx_ip, rx_port = rx_sock.getsockname()
        
        # Get actual IP address of host
        try:
            # Connect to a dummy external address to find default outgoing IP interface
            dummy = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            dummy.connect(("8.8.8.8", 80))
            my_ip = dummy.getsockname()[0]
            dummy.close()
        except Exception:
            my_ip = socket.gethostbyname(socket.gethostname())
            
        rx_sock.settimeout(timeout)
        
        # Broadcast probe
        probe = {
            "type": "DISCOVER_PROBE",
            "host_ip": my_ip,
            "host_port": rx_port
        }
        
        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        logger.info(f"Broadcasting probe from {my_ip}:{rx_port}...")
        try:
            tx_sock.sendto(json.dumps(probe).encode('utf-8'), ('255.255.255.255', DISCOVERY_PORT))
        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")
            # Try broadcasting on subnet fallback
            try:
                tx_sock.sendto(json.dumps(probe).encode('utf-8'), ('192.168.1.255', DISCOVERY_PORT))
            except Exception:
                pass
        finally:
            tx_sock.close()

        nodes = {}
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data, addr = rx_sock.recvfrom(4096)
                response = json.loads(data.decode('utf-8'))
                if response.get("type") == "DISCOVER_RESPONSE":
                    node_id = response.get("node_id")
                    if node_id:
                        # Normalize IP if it responds as localhost
                        node_ip = addr[0]
                        response["ip"] = node_ip
                        nodes[node_id] = response
                        logger.info(f"Discovered node: {node_id} at {node_ip}:{response.get('port')}")
            except socket.timeout:
                break
            except Exception as e:
                logger.error(f"Error receiving response: {e}")
                
        rx_sock.close()
        return nodes
