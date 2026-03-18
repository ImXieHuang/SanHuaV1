import tkinter as tk

class ColorPalette:
    def __init__(self, palette=None, map=None):
        self.colors = palette or [
            "#000000", "#FFFFFF", "#811AFF", "#3BFFAD", "#0000FF", "#65CCFF", "#7298FF", "#00FFFF",
            "#111111", "#222222", "#333333", "#444444", "#555555", "#666666", "#777777", "#888888",
            "#1E90FF", "#87CEEB", "#4682B4", "#5F9EA0", "#2E8B57", "#3CB371", "#20B2AA", "#48D1CC",
            "#1E90FF", "#87CEEB", "#4682B4", "#5F9EA0", "#2E8B57", "#3CB371", "#20B2AA", "#48D1CC",
            "#1E90FF", "#87CEEB", "#4682B4", "#5F9EA0", "#2E8B57", "#3CB371", "#20B2AA", "#48D1CC"
        ]
        
        self.default_map = map or {
            "background": 0,
            "text": 1,
            "border": 1,
            "highlight": 5,
            "button_bg": 12,
            "button_text": 0,
            "button_active": 17,
            "input_bg": 10,
            "input_text": 1,
            "input_cursor": 5,
            "topbar_bg": 0,
            "topbar_text": 1
        }
    
    def get_color(self, color_name):
        if color_name in self.default_map:
            return self.colors[self.default_map[color_name]]
        return self.colors[0]
    
    def get_color_by_index(self, index):
        if 0 <= index < len(self.colors):
            return self.colors[index]
        return self.colors[0]
    
    def set_color_mapping(self, mapping):
        self.default_map.update(mapping)

class InputBox:
    def __init__(self, text_string="", width=20, height=1, x=0, y=0, 
                 screen_align="center", screen_valign="middle",
                 on_submit=None, variable_name="", palette=None):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.screen_align = screen_align
        self.screen_valign = screen_valign
        self.text_string = text_string
        self.on_submit = on_submit
        self.variable_name = variable_name
        self.cursor_pos = len(text_string)
        self.active = False
        self.palette = palette or ColorPalette()
        self.dirty = True
        
        self.key_actions = {}
        self.set_default_keys()
        
        self.update_display_data()
    
    def set_default_keys(self):
        self.key_actions = {
            "Return": [self._submit_action],
            "Escape": [self._cancel_action],
            "Tab": [self._next_action],
            "BackSpace": [self._backspace_action],
            "Left": [self._move_left_action],
            "Right": [self._move_right_action],
            "Home": [self._home_action],
            "End": [self._end_action],
            "Delete": [self._delete_action]
        }
    
    def bind_key(self, key, action_name, action_func=None, append=False):
        action_to_bind = None
        if action_func:
            action_to_bind = action_func
        elif action_name == "submit":
            action_to_bind = self._submit_action
        elif action_name == "cancel":
            action_to_bind = self._cancel_action
        elif action_name == "next":
            action_to_bind = self._next_action
        elif action_name == "backspace":
            action_to_bind = self._backspace_action
        elif action_name == "left":
            action_to_bind = self._move_left_action
        elif action_name == "right":
            action_to_bind = self._move_right_action
        elif action_name == "home":
            action_to_bind = self._home_action
        elif action_name == "end":
            action_to_bind = self._end_action
        elif action_name == "delete":
            action_to_bind = self._delete_action
        
        if action_to_bind:
            if key not in self.key_actions:
                self.key_actions[key] = []
            
            if append:
                self.key_actions[key].append(action_to_bind)
            else:
                if len(self.key_actions[key]) > 0:
                    self.key_actions[key].append(action_to_bind)
                else:
                    self.key_actions[key] = [action_to_bind]
    
    def unbind_key(self, key):
        if key in self.key_actions:
            del self.key_actions[key]
    
    def _submit_action(self):
        if self.on_submit:
            self.on_submit(self.text_string, self.variable_name)
        return "submit"
    
    def _cancel_action(self):
        self.active = False
        self.dirty = True
        return True
    
    def _next_action(self):
        return "next"
    
    def _backspace_action(self):
        if self.cursor_pos > 0:
            self.text_string = self.text_string[:self.cursor_pos-1] + self.text_string[self.cursor_pos:]
            self.cursor_pos -= 1
            self.dirty = True
            return True
        return False
    
    def _move_left_action(self):
        self.cursor_pos = max(0, self.cursor_pos - 1)
        self.dirty = True
        return True
    
    def _move_right_action(self):
        self.cursor_pos = min(len(self.text_string), self.cursor_pos + 1)
        self.dirty = True
        return True
    
    def _home_action(self):
        self.cursor_pos = 0
        self.dirty = True
        return True
    
    def _end_action(self):
        self.cursor_pos = len(self.text_string)
        self.dirty = True
        return True
    
    def _delete_action(self):
        if self.cursor_pos < len(self.text_string):
            self.text_string = self.text_string[:self.cursor_pos] + self.text_string[self.cursor_pos+1:]
            self.dirty = True
            return True
        return False
    
    def set_text(self, text_string):
        self.text_string = text_string
        self.cursor_pos = len(text_string)
        self.dirty = True
    
    def update_display_data(self):
        display_text = self.text_string.ljust(self.width)[:self.width]
        
        self.text_data = []
        row = []
        for i, char in enumerate(display_text):
            is_cursor = self.active and i == self.cursor_pos
            row.append({
                "char": "|" if is_cursor else char,
                "visible": True,
                "color": self.palette.get_color("input_cursor") if is_cursor else self.palette.get_color("input_text"),
                "bg_color": self.palette.get_color("input_bg")
            })
        
        self.text_data = [row]
        self.dirty = False
    
    def set_position(self, x, y):
        self.x = x
        self.y = y
        
    def handle_input(self, char, keysym=None):
        if not self.active:
            return False
        
        result = False
        if keysym and keysym in self.key_actions:
            for action_func in self.key_actions[keysym]:
                action_result = action_func()
                if action_result:
                    result = action_result
        
        if keysym and len(keysym) == 1 and keysym in self.key_actions:
            for action_func in self.key_actions[keysym]:
                action_result = action_func()
                if action_result:
                    result = action_result
        
        if len(char) == 1 and ord(char) >= 32:
            self.text_string = self.text_string[:self.cursor_pos] + char + self.text_string[self.cursor_pos:]
            self.cursor_pos += 1
            self.dirty = True
            result = True
            
        return result
    
    def activate(self):
        self.active = True
        self.dirty = True
    
    def deactivate(self):
        self.active = False
        self.dirty = True

class Button:
    def __init__(self, text="Button", width=10, height=1, x=0, y=0,
                 screen_align="center", screen_valign="middle",
                 on_click=None, command=None, palette=None):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.screen_align = screen_align
        self.screen_valign = screen_valign
        self.text = text
        self.on_click = on_click
        self.command = command
        self.pressed = False
        self.palette = palette or ColorPalette()
        self.dirty = True
        
        self.key_actions = {}
        self.set_default_keys()
        
        self.update_display_data()
    
    def set_default_keys(self):
        self.key_actions = {
            "Return": [self._click_action],
            "space": [self._activate_action]
        }
    
    def bind_key(self, key, action_name, action_func=None, append=False):
        if action_func:
            action_to_bind = action_func
        elif action_name == "click":
            action_to_bind = self._click_action
        elif action_name == "activate":
            action_to_bind = self._activate_action
        else:
            return
        
        if key not in self.key_actions:
            self.key_actions[key] = []
        
        if append:
            self.key_actions[key].append(action_to_bind)
        else:
            if len(self.key_actions[key]) > 0:
                self.key_actions[key].append(action_to_bind)
            else:
                self.key_actions[key] = [action_to_bind]
    
    def unbind_key(self, key):
        if key in self.key_actions:
            del self.key_actions[key]
    
    def _click_action(self):
        self.handle_click()
        return True
    
    def _activate_action(self):
        self.pressed = True
        self.dirty = True
        return True
    
    def update_display_data(self):
        display_text = self.text.center(self.width)[:self.width]
        
        self.text_data = []
        row = []
        for char in display_text:
            row.append({
                "char": char,
                "visible": True,
                "color": self.palette.get_color("button_text"),
                "bg_color": self.palette.get_color("button_active") if self.pressed else self.palette.get_color("button_bg")
            })
        self.text_data = [row]
        self.dirty = False
    
    def set_position(self, x, y):
        self.x = x
        self.y = y
    
    def handle_keypress(self, keysym):
        if keysym in self.key_actions:
            for action_func in self.key_actions[keysym]:
                action_func()
            return True
        return False
    
    def handle_click(self):
        self.pressed = True
        self.dirty = True
        
        if self.on_click:
            self.on_click(self.text)
        if self.command:
            self.command()
    
    def handle_release(self):
        self.pressed = False
        self.dirty = True

class TextBox:
    def __init__(self, text_string=None, mode="single", text_align="center", text_valign="middle",
                 screen_align="center", screen_valign="middle", width=20, height=5, x=0, y=0, palette=None):
        self.mode = mode
        self.text_align = text_align
        self.text_valign = text_valign
        self.screen_align = screen_align
        self.screen_valign = screen_valign
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.palette = palette or ColorPalette()
        
        if text_string is not None:
            self.set_text(text_string)
        else:
            self.text_data = []
    
    def set_text(self, text_string):
        lines = text_string.strip().split('\n')
        self.text_data = []
        
        for line in lines:
            row = []
            line = line.strip()
            for char in line:
                row.append({
                    "char": char,
                    "visible": True,
                    "color": self.palette.get_color("text"),
                    "bg_color": self.palette.get_color("background")
                })
            self.text_data.append(row)
        
    def set_position(self, x, y):
        self.x = x
        self.y = y
        
    def set_size(self, width, height):
        self.width = width
        self.height = height
        
    def set_mode(self, mode):
        self.mode = mode
        
    def set_align(self, text_align, text_valign, screen_align, screen_valign):
        self.text_align = text_align
        self.text_valign = text_valign
        self.screen_align = screen_align
        self.screen_valign = screen_valign

class TopBar:
    def __init__(self, text_string, palette=None):
        self.text_string = text_string
        self.palette = palette or ColorPalette()
        self.set_text()
    
    def set_text(self):
        top_line = []
        for _ in range(80):
            top_line.append({
                "char": "=",
                "visible": True,
                "color": self.palette.get_color("topbar_text"),
                "bg_color": self.palette.get_color("topbar_bg")
            })
        
        title_line = []
        exit_text = "<ESC-Exit>"
        
        title_start = 0
        
        for i in range(80):
            if i >= title_start and i < title_start + len(self.text_string):
                char = self.text_string[i - title_start]
                title_line.append({
                    "char": char,
                    "visible": True,
                    "color": self.palette.get_color("topbar_text"),
                    "bg_color": self.palette.get_color("topbar_bg")
                })
            elif i >= 80 - len(exit_text):
                char = exit_text[i - (80 - len(exit_text))]
                title_line.append({
                    "char": char,
                    "visible": True,
                    "color": self.palette.get_color("highlight"),
                    "bg_color": self.palette.get_color("topbar_bg")
                })
            else:
                title_line.append({
                    "char": " ",
                    "visible": True,
                    "color": self.palette.get_color("topbar_text"),
                    "bg_color": self.palette.get_color("topbar_bg")
                })
            
        bottom_line = []
        for _ in range(80):
            bottom_line.append({
                "char": "=",
                "visible": True,
                "color": self.palette.get_color("topbar_text"),
                "bg_color": self.palette.get_color("topbar_bg")
            })
            
        self.text_data = [top_line, title_line, bottom_line]

class TUIDisplay:
    def __init__(self, root, font_path: str, palette=None):
        self.root = root
        self.font_path = font_path
        self.palette = palette or ColorPalette()
        self.screen_width = 80
        self.screen_height = 24
        self.text_boxes = []
        self.input_boxes = []
        self.buttons = []
        self.top_bars = []
        self.active_input_box = None
        self.variables = {}
        self.global_key_actions = {}
        
        self.setup_display()
        self.init_screen_content()
        self.draw_screen()

    def setup_display(self):
        self.root.title("TUI Display")
        self.root.attributes('-fullscreen', True)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.cell_width = screen_width // self.screen_width
        self.cell_height = screen_height // self.screen_height
        self.canvas = tk.Canvas(self.root, bg=self.palette.get_color("background"), highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.bind('<Key>', self.handle_keypress)
        self.root.bind('<Button-1>', self.handle_click)

    def bind_global_key(self, key, action_func, append=False):
        if key not in self.global_key_actions:
            self.global_key_actions[key] = []
        
        if append:
            self.global_key_actions[key].append(action_func)
        else:
            if len(self.global_key_actions[key]) > 0:
                self.global_key_actions[key].append(action_func)
            else:
                self.global_key_actions[key] = [action_func]
    
    def unbind_global_key(self, key):
        if key in self.global_key_actions:
            del self.global_key_actions[key]

    def handle_keypress(self, event):
        keysym = event.keysym
        
        needs_redraw = False
        
        if keysym in self.global_key_actions:
            for action_func in self.global_key_actions[keysym]:
                action_func()
        
        for input_box in self.input_boxes:
            if input_box.active:
                result = input_box.handle_input(event.char, keysym)
                if result == "next":
                    self.cycle_input_boxes()
                    needs_redraw = True
                elif result:
                    needs_redraw = True
                break
        
        for button in self.buttons:
            if button.handle_keypress(keysym):
                needs_redraw = True
                self.root.after(100, button.handle_release)

        if needs_redraw:
            self.fast_redraw()

    def handle_click(self, event):
        x = event.x // self.cell_width
        y = event.y // self.cell_height
        
        needs_redraw = False
        
        for button in self.buttons:
            button_rect = self.get_component_rect(button)
            if (button_rect[0] <= x < button_rect[2] and 
                button_rect[1] <= y < button_rect[3]):
                button.handle_click()
                needs_redraw = True
                self.root.after(100, button.handle_release)
                break
        
        for input_box in self.input_boxes:
            input_rect = self.get_component_rect(input_box)
            if (input_rect[0] <= x < input_rect[2] and 
                input_rect[1] <= y < input_rect[3]):
                if self.active_input_box:
                    self.active_input_box.deactivate()
                    needs_redraw = True
                input_box.activate()
                self.active_input_box = input_box
                needs_redraw = True
                break

        if needs_redraw:
            self.fast_redraw()

    def get_component_rect(self, component):
        screen_x = component.x
        screen_y = component.y
        
        if component.screen_align == "center":
            screen_x = (self.screen_width - component.width) // 2 + component.x
        elif component.screen_align == "right":
            screen_x = self.screen_width - component.width - component.x
            
        if component.screen_valign == "middle":
            screen_y = (self.screen_height - component.height) // 2 + component.y
        elif component.screen_valign == "bottom":
            screen_y = self.screen_height - component.height - component.y
        
        return (screen_x, screen_y, screen_x + component.width, screen_y + component.height)

    def cycle_input_boxes(self):
        if not self.input_boxes:
            return
            
        if self.active_input_box:
            current_index = self.input_boxes.index(self.active_input_box)
            next_index = (current_index + 1) % len(self.input_boxes)
        else:
            next_index = 0
            
        if self.active_input_box:
            self.active_input_box.deactivate()
            
        self.active_input_box = self.input_boxes[next_index]
        self.active_input_box.activate()
        self.fast_redraw()

    def init_screen_content(self):
        pass

    def add_text_box(self, text_box):
        self.text_boxes.append(text_box)

    def add_input_box(self, input_box):
        self.input_boxes.append(input_box)

    def add_button(self, button):
        self.buttons.append(button)

    def add_top_bar(self, top_bar):
        self.top_bars.append(top_bar)

    def set_variable(self, name, value):
        self.variables[name] = value

    def get_variable(self, name):
        return self.variables.get(name)

    def draw_component(self, component, screen_x, screen_y):
        if hasattr(component, 'update_display_data') and component.dirty:
            component.update_display_data()

        for row_idx, row in enumerate(component.text_data):
            if row_idx >= component.height:
                break
                
            for col_idx, cell_data in enumerate(row):
                if col_idx >= component.width:
                    break
                    
                actual_x = screen_x + col_idx
                actual_y = screen_y + row_idx
                
                if (0 <= actual_x < self.screen_width and 
                    0 <= actual_y < self.screen_height):
                    self.draw_cell(actual_x, actual_y, cell_data)

    def draw_cell(self, x, y, cell_data):
        if not cell_data["visible"]:
            return
            
        x1 = x * self.cell_width
        y1 = y * self.cell_height
        x2 = x1 + self.cell_width
        y2 = y1 + self.cell_height
        
        self.canvas.create_rectangle(x1, y1, x2, y2, 
                                   fill=cell_data["bg_color"], 
                                   outline=cell_data["bg_color"])
        self.canvas.create_text(x1 + self.cell_width//2, 
                              y1 + self.cell_height//2,
                              text=cell_data["char"],
                              fill=cell_data["color"],
                              font=("JetBrains Mono", self.cell_height//2, "bold"))

    def fast_redraw(self):
        self.canvas.delete("all")
        
        for top_bar in self.top_bars:
            for row_idx, row in enumerate(top_bar.text_data):
                for col_idx, cell_data in enumerate(row):
                    if col_idx < self.screen_width and row_idx < 3:
                        self.draw_cell(col_idx, row_idx, cell_data)
        
        for text_box in self.text_boxes:
            screen_x = text_box.x
            screen_y = text_box.y
            
            if text_box.screen_align == "center":
                screen_x = (self.screen_width - text_box.width) // 2 + text_box.x
            elif text_box.screen_align == "right":
                screen_x = self.screen_width - text_box.width - text_box.x
                
            if text_box.screen_valign == "middle":
                screen_y = (self.screen_height - text_box.height) // 2 + text_box.y
            elif text_box.screen_valign == "bottom":
                screen_y = self.screen_height - text_box.height - text_box.y
            
            for row_idx, row in enumerate(text_box.text_data):
                if row_idx >= text_box.height:
                    break
                    
                line_text = "".join([cell["char"] for cell in row if cell["visible"]])
                actual_text_width = len(line_text)
                
                if text_box.mode == "wrap" and actual_text_width > text_box.width:
                    line_text = line_text[:text_box.width]
                    actual_text_width = text_box.width
                
                text_x = 0
                if text_box.text_align == "center":
                    text_x = (text_box.width - actual_text_width) // 2
                elif text_box.text_align == "right":
                    text_x = text_box.width - actual_text_width
                
                text_y = row_idx
                if text_box.text_valign == "middle":
                    text_y = (text_box.height - len(text_box.text_data)) // 2 + row_idx
                elif text_box.text_valign == "bottom":
                    text_y = text_box.height - len(text_box.text_data) + row_idx
                
                for col_idx, char in enumerate(line_text):
                    if col_idx >= text_box.width:
                        break
                        
                    actual_x = screen_x + text_x + col_idx
                    actual_y = screen_y + text_y
                    
                    if (0 <= actual_x < self.screen_width and 
                        0 <= actual_y < self.screen_height and actual_y >= 3):
                        cell_data = row[col_idx] if col_idx < len(row) else row[0]
                        self.draw_cell(actual_x, actual_y, {
                            "char": char,
                            "visible": True,
                            "color": cell_data["color"],
                            "bg_color": cell_data["bg_color"]
                        })

        for input_box in self.input_boxes:
            screen_x = input_box.x
            screen_y = input_box.y
            
            if input_box.screen_align == "center":
                screen_x = (self.screen_width - input_box.width) // 2 + input_box.x
            elif input_box.screen_align == "right":
                screen_x = self.screen_width - input_box.width - input_box.x
                
            if input_box.screen_valign == "middle":
                screen_y = (self.screen_height - input_box.height) // 2 + input_box.y
            elif input_box.screen_valign == "bottom":
                screen_y = self.screen_height - input_box.height - input_box.y
            
            self.draw_component(input_box, screen_x, screen_y)

        for button in self.buttons:
            screen_x = button.x
            screen_y = button.y
            
            if button.screen_align == "center":
                screen_x = (self.screen_width - button.width) // 2 + button.x
            elif button.screen_align == "right":
                screen_x = self.screen_width - button.width - button.x
                
            if button.screen_valign == "middle":
                screen_y = (self.screen_height - button.height) // 2 + button.y
            elif button.screen_valign == "bottom":
                screen_y = self.screen_height - button.height - button.y
            
            self.draw_component(button, screen_x, screen_y)

    def draw_screen(self):
        self.fast_redraw()

    def refresh(self):
        self.fast_redraw()