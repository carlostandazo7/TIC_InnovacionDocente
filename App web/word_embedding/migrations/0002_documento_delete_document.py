# Generated by Django 4.2.6 on 2023-11-13 02:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('word_embedding', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Documento',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('archivo', models.FileField(upload_to='archivos_csv/')),
                ('contenido', models.TextField(blank=True)),
            ],
        ),
        migrations.DeleteModel(
            name='Document',
        ),
    ]
