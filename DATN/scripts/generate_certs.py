import os
import datetime
import ipaddress
import socket
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

def generate_key_and_cert(subject_name, issuer_key=None, issuer_cert=None, is_ca=False):
    # Generate ECDSA private key (SECP256R1 / prime256v1)
    private_key = ec.generate_private_key(ec.SECP256R1())
    
    # Subject name setup
    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "VN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BKU"),
        x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
    ])
    
    # Determine issuer and signing key
    if is_ca:
        issuer = subject
        signing_key = private_key
    else:
        issuer = issuer_cert.subject
        signing_key = issuer_key
    
    # Certificate builder setup
    builder = x509.CertificateBuilder()
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)
    builder = builder.public_key(private_key.public_key())
    builder = builder.serial_number(x509.random_serial_number())
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = builder.not_valid_before(now - datetime.timedelta(days=1))
    builder = builder.not_valid_after(now + datetime.timedelta(days=365))
    
    # SAN extension for server certs to prevent name mismatch errors
    if not is_ca:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName(subject_name),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                # Add local IP address dynamically to avoid validation errors
                x509.IPAddress(ipaddress.IPv4Address(socket.gethostbyname(socket.gethostname()))),
            ]),
            critical=False,
        )
        
    if is_ca:
        builder = builder.add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        
    cert = builder.sign(signing_key, hashes.SHA256())
    
    # Serialize private key
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    # Serialize certificate
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    
    return private_key, cert, key_pem, cert_pem

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    certs_dir = os.path.join(os.path.dirname(base_dir), "certs")
    os.makedirs(certs_dir, exist_ok=True)
    
    print(f"Generating certificates in: {certs_dir}")
    
    # 1. Generate Root CA
    ca_key, ca_cert, ca_key_pem, ca_cert_pem = generate_key_and_cert("OrchestRootCA", is_ca=True)
    with open(os.path.join(certs_dir, "ca.key"), "wb") as f:
        f.write(ca_key_pem)
    with open(os.path.join(certs_dir, "ca.crt"), "wb") as f:
        f.write(ca_cert_pem)
    print("  - Root CA Cert and Key generated.")
    
    # 2. Generate Host Cert (Server)
    _, _, host_key_pem, host_cert_pem = generate_key_and_cert(
        "host_master", issuer_key=ca_key, issuer_cert=ca_cert
    )
    with open(os.path.join(certs_dir, "host.key"), "wb") as f:
        f.write(host_key_pem)
    with open(os.path.join(certs_dir, "host.crt"), "wb") as f:
        f.write(host_cert_pem)
    print("  - Host Master Server Cert and Key generated.")
    
    # 3. Generate Worker Cert (Client)
    _, _, worker_key_pem, worker_cert_pem = generate_key_and_cert(
        "worker_node", issuer_key=ca_key, issuer_cert=ca_cert
    )
    with open(os.path.join(certs_dir, "worker.key"), "wb") as f:
        f.write(worker_key_pem)
    with open(os.path.join(certs_dir, "worker.crt"), "wb") as f:
        f.write(worker_cert_pem)
    print("  - Worker Client Cert and Key generated.")
    
    # 4. Generate ZeroMQ Curve Keypairs
    try:
        import zmq
        # Host Curve keypair
        h_pub, h_sec = zmq.curve_keypair()
        with open(os.path.join(certs_dir, "host_zmq.key"), "wb") as f:
            f.write(h_sec)
        with open(os.path.join(certs_dir, "host_zmq.pub"), "wb") as f:
            f.write(h_pub)
            
        # Worker Curve keypair
        w_pub, w_sec = zmq.curve_keypair()
        with open(os.path.join(certs_dir, "worker_zmq.key"), "wb") as f:
            f.write(w_sec)
        with open(os.path.join(certs_dir, "worker_zmq.pub"), "wb") as f:
            f.write(w_pub)
        print("  - ZeroMQ Curve Keypairs generated.")
    except Exception as e:
        print(f"  - Warning: Failed to generate ZeroMQ Curve keypairs: {e}")
        
    print("\n[SUCCESS] Cryptographic credentials created successfully!")

if __name__ == "__main__":
    main()
