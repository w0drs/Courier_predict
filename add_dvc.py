import yaml
import json
import os

# Читаем dvc.lock
with open('dvc.lock', 'r', encoding='utf-8') as f:
    lock_data = yaml.safe_load(f)


# Функция для создания .dvc файла
def create_dvc_file(filepath, md5_hash, size):
    dvc_content = {
        'outs': [{
            'path': os.path.basename(filepath),
            'md5': md5_hash,
            'size': size
        }]
    }

    dvc_filename = filepath + '.dvc'
    os.makedirs(os.path.dirname(dvc_filename), exist_ok=True)

    with open(dvc_filename, 'w', encoding='utf-8') as f:
        json.dump(dvc_content, f, indent=2)

    print(f"✓ Created: {dvc_filename}")


# Создаем .dvc файлы для всех outs
for stage_name, stage_data in lock_data['stages'].items():
    if 'outs' in stage_data:
        for out_file in stage_data['outs']:
            create_dvc_file(
                out_file['path'],
                out_file['md5'],
                out_file['size']
            )

print("\n✅ Все .dvc файлы созданы!")