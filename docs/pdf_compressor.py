import os
import sys
from PIL import Image
import io
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    from PyPDF2 import PdfFileReader as PdfReader, PdfFileWriter as PdfWriter
import fitz  # PyMuPDF
import argparse
from pathlib import Path

class PDFCompressor:
    def __init__(self, input_path, output_path=None, quality=85):
        """
        Inicializa el compresor de PDF
        
        Args:
            input_path: Ruta del PDF de entrada
            output_path: Ruta del PDF de salida (opcional)
            quality: Calidad de compresión de imágenes (1-100)
        """
        self.input_path = input_path
        self.output_path = output_path or self._generate_output_path()
        self.quality = quality
        
    def _generate_output_path(self):
        """Genera un nombre de archivo de salida basado en el original"""
        path = Path(self.input_path)
        return str(path.parent / f"{path.stem}_compressed{path.suffix}")
    
    def compress_with_pymupdf(self):
        """
        Comprime el PDF usando PyMuPDF (método más efectivo)
        """
        try:
            # Abrir el PDF original
            pdf_document = fitz.open(self.input_path)
            
            # Crear un nuevo PDF para la versión comprimida
            pdf_output = fitz.open()
            
            # Calcular factor de escala basado en la calidad
            scale = self.quality / 100.0
            if scale < 0.5:
                scale = 0.5  # Limitar la reducción mínima
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Obtener el contenido de la página como pixmap con escala reducida
                mat = fitz.Matrix(scale, scale)  # Reducir resolución
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convertir a imagen PIL para comprimir
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                
                # Comprimir la imagen
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=int(self.quality * 0.8), optimize=True)
                img_buffer.seek(0)
                
                # Crear una nueva página con la imagen comprimida
                rect = page.rect
                new_page = pdf_output.new_page(width=rect.width, height=rect.height)
                new_page.insert_image(rect, stream=img_buffer.getvalue())
                
                pix = None  # Liberar memoria
            
            # Guardar con opciones de compresión máximas
            save_options = {
                "garbage": 4,  # Eliminar objetos no utilizados
                "deflate": True,  # Comprimir streams
                "clean": True,  # Limpiar contenido
                "deflate_images": True,  # Comprimir imágenes
                "deflate_fonts": True,  # Comprimir fuentes
            }
            
            # Intentar guardar con opciones avanzadas si están disponibles
            try:
                pdf_output.save(self.output_path, **save_options)
            except TypeError:
                # Si falla, guardar sin opciones avanzadas
                pdf_output.save(self.output_path, garbage=4, deflate=True)
            
            # Cerrar documentos
            pdf_output.close()
            pdf_document.close()
            
            # Calcular y mostrar la reducción de tamaño
            original_size = os.path.getsize(self.input_path)
            compressed_size = os.path.getsize(self.output_path)
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"✓ Compresión completada")
            print(f"  Archivo original: {self._format_size(original_size)}")
            print(f"  Archivo comprimido: {self._format_size(compressed_size)}")
            print(f"  Reducción: {reduction:.1f}%")
            print(f"  Guardado en: {self.output_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error durante la compresión con PyMuPDF: {str(e)}")
            return False
    
    def compress_with_pypdf2(self):
        """
        Método alternativo usando PyPDF2 (menos efectivo pero más compatible)
        """
        try:
            # Intentar abrir el PDF
            with open(self.input_path, 'rb') as file:
                reader = PdfReader(file)
                writer = PdfWriter()
                
                # Procesar cada página
                for i, page in enumerate(reader.pages):
                    # Comprimir contenido de la página si el método existe
                    if hasattr(page, 'compress_content_streams'):
                        page.compress_content_streams()
                    elif hasattr(page, 'compressContentStreams'):
                        page.compressContentStreams()
                    
                    writer.add_page(page)
                
                # Eliminar duplicados si el método existe
                if hasattr(writer, 'compress_identical_objects'):
                    try:
                        writer.compress_identical_objects(remove_use_none=True)
                    except:
                        pass
                
                # Guardar el PDF comprimido
                with open(self.output_path, 'wb') as output_file:
                    writer.write(output_file)
            
            # Calcular reducción
            original_size = os.path.getsize(self.input_path)
            compressed_size = os.path.getsize(self.output_path)
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"✓ Compresión completada (método PyPDF2)")
            print(f"  Archivo original: {self._format_size(original_size)}")
            print(f"  Archivo comprimido: {self._format_size(compressed_size)}")
            print(f"  Reducción: {reduction:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"✗ Error con PyPDF2: {str(e)}")
            return False
    
    def compress_moderate(self):
        """
        Compresión moderada que mantiene la estructura del PDF
        """
        try:
            pdf_document = fitz.open(self.input_path)
            
            # Reducir resolución de imágenes según la calidad
            if self.quality < 70:
                max_dim = 1200  # Baja calidad
            elif self.quality < 85:
                max_dim = 1600  # Media calidad
            else:
                max_dim = 2400  # Alta calidad
            
            modified = False
            
            # Procesar cada página
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                if not image_list:
                    continue
                    
                # Procesar cada imagen
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        # Solo procesar si la imagen es grande
                        if pix.width > max_dim or pix.height > max_dim:
                            # Calcular nuevo tamaño
                            scale = min(max_dim / pix.width, max_dim / pix.height)
                            mat = fitz.Matrix(scale, scale)
                            pix = fitz.Pixmap(pix, mat)
                            modified = True
                        
                        # Convertir a RGB si es necesario
                        if pix.alpha:
                            pix2 = fitz.Pixmap(fitz.csRGB, pix)
                            pix = pix2
                        
                        # Comprimir según calidad
                        jpg_quality = int(self.quality * 0.9)
                        img_data = pix.tobytes("jpeg", jpg_quality=jpg_quality)
                        
                        # Solo actualizar si la nueva imagen es menor
                        if len(img_data) < len(pix.tobytes()):
                            pdf_document._updateObject(xref, stream=img_data)
                            modified = True
                        
                        pix = None
                    except:
                        continue
            
            if not modified:
                print("  No se encontraron imágenes grandes para comprimir")
                pdf_document.close()
                # Intentar solo optimización básica
                return self.optimize_pdf_structure()
            
            # Guardar con opciones moderadas
            pdf_document.save(
                self.output_path,
                garbage=3,  # Limpieza moderada
                deflate=True,
                clean=True
            )
            pdf_document.close()
            
            # Mostrar resultados
            original_size = os.path.getsize(self.input_path)
            compressed_size = os.path.getsize(self.output_path)
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"✓ Compresión moderada completada")
            print(f"  Archivo original: {self._format_size(original_size)}")
            print(f"  Archivo comprimido: {self._format_size(compressed_size)}")
            print(f"  Reducción: {reduction:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"✗ Error en compresión moderada: {str(e)}")
            return False
    
    def optimize_pdf_structure(self):
        """
        Solo optimiza la estructura del PDF sin tocar las imágenes
        """
        try:
            pdf_document = fitz.open(self.input_path)
            
            # Guardar con optimización básica
            pdf_document.save(
                self.output_path,
                garbage=4,      # Máxima limpieza de objetos no usados
                deflate=True,   # Comprimir streams
                clean=True,     # Limpiar contenido
                pretty=False,   # No formatear
                ascii=False,    # Permitir caracteres no ASCII
                expand=0        # No expandir
            )
            pdf_document.close()
            
            # Resultados
            original_size = os.path.getsize(self.input_path)
            compressed_size = os.path.getsize(self.output_path)
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"✓ Optimización estructural completada")
            print(f"  Archivo original: {self._format_size(original_size)}")
            print(f"  Archivo comprimido: {self._format_size(compressed_size)}")
            print(f"  Reducción: {reduction:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"✗ Error en optimización: {str(e)}")
            return False

    def compress_images_method(self):
        """
        Método que extrae y comprime solo las imágenes del PDF
        """
        try:
            pdf_document = fitz.open(self.input_path)
            
            # Contar imágenes totales
            total_images = 0
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                total_images += len(image_list)
            
            if total_images == 0:
                print("  No se encontraron imágenes en el PDF")
                pdf_document.close()
                # Si no hay imágenes, intentar comprimir el PDF completo como imagen
                return self.compress_with_pymupdf()
            
            print(f"  Procesando {total_images} imágenes...")
            
            # Procesar cada página
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                # Procesar cada imagen
                for img_index, img in enumerate(image_list):
                    try:
                        # Extraer la imagen
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        # Si la imagen es muy grande, reducir su tamaño
                        if pix.width > 1920 or pix.height > 1920:
                            # Calcular factor de escala
                            scale = min(1920.0 / pix.width, 1920.0 / pix.height)
                            mat = fitz.Matrix(scale, scale)
                            pix = fitz.Pixmap(pix, mat)
                        
                        # Convertir a RGB si es necesario
                        if pix.alpha:
                            pix2 = fitz.Pixmap(fitz.csRGB, pix)
                            pix = pix2
                        
                        # Comprimir como JPEG con calidad reducida
                        img_data = pix.tobytes("jpeg", jpg_quality=int(self.quality * 0.7))
                        
                        # Reemplazar la imagen
                        pdf_document._updateObject(xref, stream=img_data)
                        
                        pix = None
                    except:
                        continue
            
            # Guardar el documento con máxima compresión
            pdf_document.save(self.output_path, garbage=4, deflate=True, clean=True)
            pdf_document.close()
            
            # Mostrar resultados
            original_size = os.path.getsize(self.input_path)
            compressed_size = os.path.getsize(self.output_path)
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"✓ Compresión de imágenes completada")
            print(f"  Archivo original: {self._format_size(original_size)}")
            print(f"  Archivo comprimido: {self._format_size(compressed_size)}")
            print(f"  Reducción: {reduction:.1f}%")
            
            # Si no hubo reducción significativa, intentar método más agresivo
            if reduction < 5:
                print("  ⚠ Reducción mínima, intentando compresión más agresiva...")
                return self.compress_with_pymupdf()
            
            return True
            
        except Exception as e:
            print(f"✗ Error en compresión de imágenes: {str(e)}")
            return False
    
    def _format_size(self, size_bytes):
        """Formatea el tamaño en bytes a una unidad legible"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def compress(self, method='auto'):
        """
        Comprime el PDF usando el método especificado
        
        Args:
            method: 'pymupdf', 'pypdf2', 'images', o 'auto' (intenta varios métodos)
        """
        if not os.path.exists(self.input_path):
            print(f"✗ Error: No se encontró el archivo {self.input_path}")
            return False
        
        print(f"Comprimiendo PDF: {self.input_path}")
        print(f"Calidad de imagen: {self.quality}%")
        
        methods = {
            'pymupdf': self.compress_with_pymupdf,
            'pypdf2': self.compress_with_pypdf2,
            'images': self.compress_images_method,
            'moderate': self.compress_moderate,
            'optimize': self.optimize_pdf_structure
        }
        
        if method in methods:
            return methods[method]()
        
        # Modo auto: intentar varios métodos
        if method == 'auto':
            # Primero intentar compresión moderada
            if self.compress_moderate():
                return True
            
            # Si no es suficiente, intentar comprimir solo imágenes
            if self.compress_images_method():
                return True
            
            # Si falla, intentar PyMuPDF completo
            if self.compress_with_pymupdf():
                return True
            
            # Último recurso: PyPDF2
            return self.compress_with_pypdf2()
        
        return False


def batch_compress(pattern, quality=85, method='auto'):
    """
    Comprime múltiples PDFs que coincidan con un patrón
    
    Args:
        pattern: Patrón de archivos (ej: "*.pdf")
        quality: Calidad de compresión
        method: Método a usar
    """
    from glob import glob
    
    files = glob(pattern)
    if not files:
        print(f"No se encontraron archivos que coincidan con: {pattern}")
        return
    
    print(f"Encontrados {len(files)} archivos PDF")
    
    success_count = 0
    for file_path in files:
        print(f"\n{'='*50}")
        compressor = PDFCompressor(file_path, quality=quality)
        if compressor.compress(method):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Compresión completada: {success_count}/{len(files)} archivos procesados")


def find_optimal_quality(input_path, target_size_mb, method='auto'):
    """
    Encuentra la calidad óptima para alcanzar un tamaño objetivo
    
    Args:
        input_path: Ruta del PDF
        target_size_mb: Tamaño objetivo en MB
        method: Método de compresión
    """
    target_size_bytes = target_size_mb * 1024 * 1024
    original_size = os.path.getsize(input_path)
    
    print(f"\nBuscando calidad óptima para alcanzar ~{target_size_mb} MB...")
    print(f"Tamaño original: {original_size / 1024 / 1024:.2f} MB")
    
    # Búsqueda binaria de la calidad óptima
    low_quality = 1
    high_quality = 100
    best_quality = 85
    best_size = 0
    attempts = 0
    
    while low_quality <= high_quality and attempts < 10:
        current_quality = (low_quality + high_quality) // 2
        temp_output = input_path.replace('.pdf', f'_temp_q{current_quality}.pdf')
        
        print(f"\nProbando calidad {current_quality}%...")
        compressor = PDFCompressor(input_path, temp_output, current_quality)
        
        if compressor.compress(method):
            current_size = os.path.getsize(temp_output)
            print(f"  → Tamaño resultante: {current_size / 1024 / 1024:.2f} MB")
            
            # Si está dentro del 10% del objetivo, es aceptable
            if abs(current_size - target_size_bytes) <= target_size_bytes * 0.1:
                best_quality = current_quality
                best_size = current_size
                os.remove(temp_output)
                break
            
            # Ajustar búsqueda
            if current_size > target_size_bytes:
                # Archivo muy grande, reducir calidad
                high_quality = current_quality - 1
            else:
                # Archivo muy pequeño, aumentar calidad
                low_quality = current_quality + 1
                best_quality = current_quality
                best_size = current_size
            
            os.remove(temp_output)
        
        attempts += 1
    
    print(f"\n{'='*50}")
    print(f"✓ Calidad óptima encontrada: {best_quality}%")
    print(f"  Tamaño esperado: ~{best_size / 1024 / 1024:.2f} MB")
    return best_quality


def main():
    parser = argparse.ArgumentParser(
        description='Comprime archivos PDF reduciendo la calidad de las imágenes'
    )
    parser.add_argument('input', help='Ruta del archivo PDF o patrón para múltiples archivos')
    parser.add_argument('-o', '--output', help='Ruta del archivo PDF de salida')
    parser.add_argument(
        '-q', '--quality', 
        type=int, 
        default=85,
        help='Calidad de compresión de imágenes (1-100, default: 85)'
    )
    parser.add_argument(
        '-s', '--size',
        type=float,
        help='Tamaño objetivo en MB (busca la calidad óptima automáticamente)'
    )
    parser.add_argument(
        '-m', '--method',
        choices=['pymupdf', 'pypdf2', 'images', 'moderate', 'optimize', 'auto'],
        default='auto',
        help='Método de compresión a usar (default: auto)'
    )
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='Procesar múltiples archivos (usar con patrón como "*.pdf")'
    )
    
    args = parser.parse_args()
    
    # Si se especifica tamaño objetivo, encontrar calidad óptima
    if args.size:
        if args.batch:
            print("✗ Error: No se puede usar --size con --batch")
            sys.exit(1)
        
        optimal_quality = find_optimal_quality(args.input, args.size, args.method)
        compressor = PDFCompressor(args.input, args.output, optimal_quality)
        success = compressor.compress(args.method)
        sys.exit(0 if success else 1)
    
    # Validar calidad
    if not 1 <= args.quality <= 100:
        print("✗ Error: La calidad debe estar entre 1 y 100")
        sys.exit(1)
    
    # Modo batch
    if args.batch:
        batch_compress(args.input, args.quality, args.method)
    else:
        # Crear compresor y ejecutar
        compressor = PDFCompressor(args.input, args.output, args.quality)
        success = compressor.compress(args.method)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


# Ejemplos de uso:
# python pdf_compressor.py documento.pdf
# python pdf_compressor.py documento.pdf -o documento_pequeno.pdf -q 70
# python pdf_compressor.py documento.pdf --method images
# python pdf_compressor.py "*.pdf" --batch -q 80