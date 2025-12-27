from google.cloud import storage

client = storage.Client()
print(f"Authenticated as project: {client.project}")