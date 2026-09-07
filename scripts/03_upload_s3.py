import boto3
import os
from botocore.exceptions import ClientError

# Configuração de Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dados_sinteticos_ifms.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_evasao_ifms.joblib')

BUCKET_NAME = 'evasao-zero-dados-ifms-katy-849275'

def criar_bucket_seguro(s3_client, bucket_name):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' já existe e está acessível.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"🔄 Criando bucket '{bucket_name}'...")
            s3_client.create_bucket(Bucket=bucket_name)
            
            # Aplica o bloqueio de acesso público (Requisito de Segurança Trello)
            s3_client.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': True,
                    'IgnorePublicAcls': True,
                    'BlockPublicPolicy': True,
                    'RestrictPublicBuckets': True
                }
            )
            print("✅ Bucket criado e acesso público bloqueado com sucesso.")
        else:
            print(f"❌ Erro ao acessar bucket: {e}")
            raise

def upload_artefato(s3_client, caminho_local, bucket, chave_s3):
    try:
        s3_client.upload_file(caminho_local, bucket, chave_s3)
        print(f"📤 Upload concluído: {chave_s3}")
    except ClientError as e:
        print(f"❌ Falha no upload de {chave_s3}: {e}")

def main():
    print("=" * 60)
    print(" INTEGRAÇÃO AWS S3 - EVASÃO ZERO")
    print("=" * 60)
    
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    criar_bucket_seguro(s3_client, BUCKET_NAME)
    
    print("\nIniciando migração de artefatos...")
    upload_artefato(s3_client, DATA_PATH, BUCKET_NAME, 'dados/dados_sinteticos_ifms.csv')
    upload_artefato(s3_client, MODEL_PATH, BUCKET_NAME, 'modelos/modelo_evasao_ifms.joblib')
    
    print("\n============================================================")
    print("[SUCESSO] Infraestrutura de dados sincronizada com a nuvem.")
    print("============================================================")

if __name__ == '__main__':
    main()