import struct
import tkinter as tk

class TTFRenderer:
    def __init__(self, font_path):
        self.font_path = font_path
        self.char_images = {}
        self.cell_width = 8
        self.cell_height = 16
        self.load_font()
    
    def load_font(self):
        with open(self.font_path, 'rb') as f:
            self.data = f.read()
        
        self.numTables = struct.unpack('>H', self.data[4:6])[0]
        
        self.tables = {}
        for i in range(self.numTables):
            offset = 12 + i * 16
            tag = self.data[offset:offset+4].decode('ascii')
            table_offset = struct.unpack('>I', self.data[offset+8:offset+12])[0]
            table_length = struct.unpack('>I', self.data[offset+12:offset+16])[0]
            self.tables[tag] = (table_offset, table_length)
        
        self.cmap = self.parse_cmap()
        
        self.loca = self.parse_loca()
        self.glyf_offset = self.tables['glyf'][0]
        
        for i in range(32, 127):
            self.char_images[chr(i)] = self.render_char(chr(i))
    
    def parse_cmap(self):
        offset, length = self.tables['cmap']
        cmap_data = self.data[offset:offset+length]
        numTables = struct.unpack('>H', cmap_data[2:4])[0]
        for i in range(numTables):
            platform = struct.unpack('>H', cmap_data[4 + i*8:4 + i*8+2])[0]
            encoding = struct.unpack('>H', cmap_data[4 + i*8+2:4 + i*8+4])[0]
            if platform == 3 and encoding == 1:
                sub_offset = struct.unpack('>I', cmap_data[4 + i*8+4:4 + i*8+8])[0]
                return self.parse_cmap_subtable(cmap_data[sub_offset:])
        return {}
    
    def parse_cmap_subtable(self, sub_data):
        format = struct.unpack('>H', sub_data[0:2])[0]
        if format == 4:
            length = struct.unpack('>H', sub_data[2:4])[0]
            segCount = struct.unpack('>H', sub_data[6:8])[0] // 2
            endCodes = [struct.unpack('>H', sub_data[14 + i*2:16 + i*2])[0] for i in range(segCount)]
            startCodes = [struct.unpack('>H', sub_data[16 + segCount*2 + i*2:18 + segCount*2 + i*2])[0] for i in range(segCount)]
            idDeltas = [struct.unpack('>H', sub_data[16 + segCount*4 + i*2:18 + segCount*4 + i*2])[0] for i in range(segCount)]
            idRangeOffsets = [struct.unpack('>H', sub_data[16 + segCount*6 + i*2:18 + segCount*6 + i*2])[0] for i in range(segCount)]
            
            cmap = {}
            for seg in range(segCount):
                for code in range(startCodes[seg], endCodes[seg] + 1):
                    if idRangeOffsets[seg] == 0:
                        glyph_index = (code + idDeltas[seg]) % 65536
                    else:
                        glyph_index = 0
                    cmap[code] = glyph_index
            return cmap
        return {}
    
    def parse_loca(self):
        offset, length = self.tables['loca']
        loca_data = self.data[offset:offset+length]
        numGlyphs = self.tables['maxp'][1] // 2
        loca = []
        for i in range(numGlyphs + 1):
            loca.append(struct.unpack('>H', loca_data[i*2:(i+1)*2])[0] * 2)
        return loca
    
    def render_char(self, char):
        code = ord(char)
        glyph_index = self.cmap.get(code, 0)
        if glyph_index >= len(self.loca) - 1:
            glyph_index = 0
        
        glyph_offset = self.glyf_offset + self.loca[glyph_index]
        glyph_length = self.loca[glyph_index + 1] - self.loca[glyph_index]
        glyph_data = self.data[glyph_offset:glyph_offset + glyph_length]
        
        img = tk.PhotoImage(width=self.cell_width, height=self.cell_height)
        pixels = ["#FFFFFF"] * (self.cell_width * self.cell_height)
        if glyph_length > 0:
            pixels[self.cell_width * (self.cell_height // 2) + self.cell_width // 2] = "#000000"
        img.put(" ".join(pixels))
        return img
    
    def get_char_image(self, char):
        return self.char_images.get(char, self.char_images.get(' ', None))