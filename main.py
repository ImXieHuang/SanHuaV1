import tkinter as tk
from pathlib import Path
from Outputio.tuidisplay import TUIDisplay, TopBar, TextBox, InputBox, Button, ColorPalette

def main():
    root = tk.Tk()
    font_path = Path(__file__).parent / "Outputio" / "fonts" / "JetBrainsMono-Bold.ttf"
    
    if font_path.exists():
        palette = ColorPalette()
        
        app = TUIDisplay(root, str(font_path), palette)
        
        def quit_app():
            root.quit()
        
        def toggle_fullscreen():
            root.attributes('-fullscreen', not root.attributes('-fullscreen'))

        def handle_submit(text, variable_name):  
            inputs.text_string = ""
            inputs.cursor_pos = 0
            textprocessor(text)
            
            inputs.dirty = True
            app.fast_redraw()
        
        app.bind_global_key("Escape", quit_app)
        app.bind_global_key("F11", toggle_fullscreen)
        
        top_bar = TopBar("-SanHuaV1 TUI-", palette)
        app.add_top_bar(top_bar)
        
        img1 = '''
' ∧ _ ∧   +
/ . _ . \\  
≡ _   ∧ ≡ 
` ∧ | / \\ /
( _ w _ _ )
'''

        box = TextBox(img1, text_align="left", x=3, palette=palette)
        app.add_text_box(box)

        inputs = InputBox(width=45, x=-2, y=10, on_submit=handle_submit)
        app.add_input_box(inputs)

        enter = TextBox("[<-]", y=10, x=22, palette=ColorPalette(map={"text": 1, "background": 5}))
        app.add_text_box(enter)

        def update():
            app.refresh()
            root.after(50, update)
            
        root.after(50, update)
        root.mainloop()
    else:
        print("字体文件未找到，请确保 'JetBrainsMono-Bold.ttf' 在 'fonts' 目录中")

def textprocessor(text):
    print(text)

if __name__ == "__main__":
    main()

