# LR Parser Uygulaması

**Öğrenci Bilgileri:**
- Ad Soyad: Arda Timuçin ACAR, İbrahim YILMAZ
- Numara: B221202013, B221202043

## Proje Açıklaması

Bu proje, SWE 204 - Concepts of Programming Languages dersi için hazırlanmış bir LR parser uygulamasıdır. Uygulama, temel aritmetik ifadeleri ayrıştırmak için bir grammar, aksiyon tablosu ve goto tablosu kullanır. Her bir girdi ifadesi için bir izleme tablosu ve ayrıştırma ağacı üretir.

## Kullanılan Teknolojiler ve Araçlar

- **Programlama Dili:** Python 3
- **Veri Yapıları:** 
  - LR Parser için durum yığını (stack)
  - Sembol yığını
  - Ayrıştırma ağacı için ağaç veri yapısı
  - İzleme tablosu için liste veri yapısı

## Proje Yapısı

Proje aşağıdaki dosyalardan oluşmaktadır:

- `lr_parser.py`: Ana LR parser uygulaması
- `test_script.py`: Tüm giriş dosyalarını test etmek için yardımcı script

## Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.6 veya üzeri

### Dosya Yapısı

Uygulamanın çalışması için aşağıdaki dosyaların aynı dizinde bulunması gerekmektedir:

1. `lr_parser.py`
2. `Grammar.txt`
3. `ActionTable.txt`
4. `GotoTable.txt`
5. `input1.txt`, `input2.txt`, ..., `input9.txt`

### Çalıştırma

Uygulamayı çalıştırmak için:

```bash
python lr_parser.py input1.txt
```

Bu komut, `input1.txt` dosyasındaki ifadeyi ayrıştırır ve sonucu `output1.txt` dosyasına kaydeder.

Tüm giriş dosyalarını test etmek için:

```bash
python test_script.py
```

## Uygulama Detayları

### LR Parser İşleyişi

1. Uygulama, `Grammar.txt`, `ActionTable.txt` ve `GotoTable.txt` dosyalarını okur.
2. Giriş ifadesindeki tokenlar ayrıştırılır.
3. LR parsing algoritması çalıştırılır:
   - Durum yığını, sembol yığını ve düğüm yığını oluşturulur.
   - Aksiyon tablosuna göre shift, reduce ve accept işlemleri gerçekleştirilir.
   - Her adımda izleme tablosu güncellenir.
4. Ayrıştırma başarılı ise ayrıştırma ağacı oluşturulur.
5. İzleme tablosu ve ayrıştırma ağacı çıktı dosyasına yazdırılır.

### Sınıf Yapısı

- `ActionType`: LR parser eylem tiplerini temsil eden enum sınıfı (shift, reduce, accept, error)
- `Action`: LR parser eylemlerini temsil eden sınıf
- `Rule`: Gramer kurallarını temsil eden sınıf
- `TreeNode`: Ayrıştırma ağacı düğümlerini temsil eden sınıf
- `LRParser`: Ana LR parser sınıfı

### Hata İşleme

Uygulama aşağıdaki durumlarda hata veriyor ve izleme tablosuna kaydediyor:

1. Geçersiz token durumları
2. Sözdizimi hataları
3. Goto tablosunda olmayan durumlar
4. Bilinmeyen eylemler

## Test Sonuçları

Uygulama, verilen tüm giriş dosyaları için test edilmiştir. Sonuçlar aşağıdaki gibidir:

- `input1.txt` - `input6.txt` ve `input8.txt`: Başarılı ayrıştırma
- `input7.txt`: Sözdizimi hatası (syntax error)
- `input9.txt`: Bilinmeyen token hatası (unknown token)

## Örnek Çıktı

İşte `input1.txt` dosyası için örnek bir çıktı:

```
PARSE TRACING TABLE:
Stack | Input | Action
--------------------------------------------------
0 | id + id * id $ | s5
0 5 | + id * id $ | r6
0 3 | + id * id $ | r4
0 2 | + id * id $ | s6
0 2 6 | id * id $ | s5
0 2 6 5 | * id $ | r6
0 2 6 3 | * id $ | r4
0 2 6 9 | * id $ | s7
0 2 6 9 7 | id $ | s5
0 2 6 9 7 5 | $ | r6
0 2 6 9 7 10 | $ | r3
0 2 6 9 | $ | r1
0 1 | $ | acc

PARSE TREE:
E
  E
    T
      F
        id
  +
  T
    T
      F
        id
    *
    F
      id
```

## Notlar ve Kaynaklar

- LR parser algoritması için "Concepts of Programming Languages" ders materyalleri kullanılmıştır.
- Python'un veri yapıları ve Enum modülü hakkında bilgi için Python resmi dokümantasyonu referans alınmıştır.