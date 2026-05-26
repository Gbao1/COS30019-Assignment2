from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk

from tbrgs.config_loader import load_config
from tbrgs.network_builder import (
    RoadGraph,
    build_knn_road_graph,
    load_scats_sites,
    load_scats_sites_from_traffic_workbook,
)
from tbrgs.pipeline import TBRGSContext, build_context, recommend_routes


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "tbrgs_defaults.json"
DEFAULT_PICK_RADIUS_PX = 22.0
DEFAULT_MAP_ZOOM = 1.35
MIN_MAP_ZOOM = 0.7
MAX_MAP_ZOOM = 3.0
ZOOM_STEP = 1.12
DEFAULT_MIN_MAP_SIZE = (1100, 800)


class TBRGSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TBRGS - Assignment 2B")
        self.root.geometry("1200x700")

        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.ctx: TBRGSContext | None = None
        self._ctx_lock = threading.Lock()
        self.is_busy = False
        self._route_colors = ["#d7191c", "#2c7bb6", "#1a9641", "#fdae61", "#984ea3"]
        self._route_cache: dict[tuple[int, int, int, str, str, int], list] = {}
        self._current_routes = None
        self._hover_route_idx: int | None = None
        self._current_route_count = 0
        self._is_dragging_map = False
        self._pressed_origin_node: int | None = None
        self.graph = self._load_graph_preview()
        self.pick_radius_px = DEFAULT_PICK_RADIUS_PX
        self.map_padding_px = (45, 45, 45, 45)
        self.map_zoom = DEFAULT_MAP_ZOOM
        self.min_map_width, self.min_map_height = DEFAULT_MIN_MAP_SIZE

        runtime = self.config["runtime"]

        form = ttk.Frame(root, padding=12)
        form.pack(fill=tk.X)

        ttk.Label(form, text="Origin SCATS").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.origin_var = tk.StringVar(value=str(runtime["default_origin"]))
        ttk.Entry(form, textvariable=self.origin_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Destination SCATS").grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)
        self.destination_var = tk.StringVar(value=str(runtime["default_destination"]))
        ttk.Entry(form, textvariable=self.destination_var, width=12).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Top-k").grid(row=0, column=4, sticky=tk.W, padx=4, pady=4)
        self.topk_var = tk.StringVar(value=str(runtime["default_top_k"]))
        ttk.Entry(form, textvariable=self.topk_var, width=8).grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Model").grid(row=0, column=6, sticky=tk.W, padx=4, pady=4)
        self.model_var = tk.StringVar(value=runtime["default_model"])
        ttk.Combobox(
            form,
            textvariable=self.model_var,
            values=["best", "lstm", "gru", "rf"],
            state="readonly",
            width=10,
        ).grid(row=0, column=7, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Algorithm").grid(row=0, column=8, sticky=tk.W, padx=4, pady=4)
        self.algorithm_var = tk.StringVar(value="CUS2")
        self.algorithm_var.set(str(runtime.get("default_algorithm", "CUS2")))
        ttk.Combobox(
            form,
            textvariable=self.algorithm_var,
            values=["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"],
            state="readonly",
            width=8,
        ).grid(row=0, column=9, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Hour").grid(row=0, column=10, sticky=tk.W, padx=4, pady=4)
        self.hour_var = tk.StringVar(value="9")
        self.hour_var.set(str(runtime.get("default_hour", 9)))
        ttk.Spinbox(form, from_=0, to=23, textvariable=self.hour_var, width=5).grid(
            row=0,
            column=11,
            sticky=tk.W,
            padx=4,
            pady=4,
        )

        self.run_button = ttk.Button(form, text="Find Path", command=self.on_run)
        self.run_button.grid(row=0, column=12, sticky=tk.W, padx=8, pady=4)

        self.zoom_var = tk.StringVar(value=f"Zoom: {self.map_zoom:.2f}x")
        ttk.Label(form, textvariable=self.zoom_var, foreground="#444").grid(
            row=0,
            column=13,
            sticky=tk.W,
            padx=(8, 4),
            pady=4,
        )

        self.status_var = tk.StringVar(value="Ready. Left-click sets Origin, Right-click sets Destination.")
        ttk.Label(form, textvariable=self.status_var, foreground="#2b2b2b").grid(
            row=1,
            column=0,
            columnspan=13,
            sticky=tk.W,
            padx=4,
            pady=(2, 0),
        )

        ttk.Label(
            form,
            text="Map Instruction: Left-click=Origin, Right-click=Destination, Mouse wheel=Zoom, then Find Path.",
            foreground="#0b5aa2",
        ).grid(
            row=2,
            column=0,
            columnspan=13,
            sticky=tk.W,
            padx=4,
            pady=(2, 0),
        )

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.main_pane = main
        self._default_split_applied = False

        text_frame = ttk.Frame(main)
        viz_frame = ttk.Frame(main)
        # Default split: display/output 30%, map 70%.
        main.add(text_frame, weight=3)
        main.add(viz_frame, weight=7)

        self.output = tk.Text(text_frame, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

        canvas_wrap = ttk.Frame(viz_frame)
        canvas_wrap.pack(fill=tk.BOTH, expand=True)

        x_scroll = ttk.Scrollbar(canvas_wrap, orient=tk.HORIZONTAL)
        y_scroll = ttk.Scrollbar(canvas_wrap, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(
            canvas_wrap,
            bg="white",
            highlightthickness=1,
            highlightbackground="#d0d0d0",
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
        )
        x_scroll.configure(command=self.canvas.xview)
        y_scroll.configure(command=self.canvas.yview)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_map_right_click)
        self.canvas.bind("<MouseWheel>", self.on_map_wheel)
        self.canvas.bind("<Button-4>", self.on_map_wheel)
        self.canvas.bind("<Button-5>", self.on_map_wheel)

        # Force initial pane split because weight-only sizing can vary by platform/theme.
        self.main_pane.bind("<Configure>", self._ensure_default_split)

        self._draw_base_graph()
        self.output.insert(tk.END, "Ready. Left-click to set origin, right-click to set destination, then click Find Path.\n")
        self.root.after(150, self._ensure_default_split)

    def _load_graph_preview(self) -> RoadGraph:
        data_cfg = self.config["data"]
        net_cfg = self.config["network"]

        try:
            sites_df = load_scats_sites_from_traffic_workbook(data_cfg["traffic_file"])
        except Exception:
            sites_df = load_scats_sites(
                site_file=data_cfg["site_file"],
                fallback_file=data_cfg["location_fallback_file"],
            )

        return build_knn_road_graph(
            sites_df=sites_df,
            k_neighbors=int(net_cfg["k_nearest_neighbors"]),
            max_neighbor_distance_km=float(net_cfg["max_neighbor_distance_km"]),
        )

    def _set_busy(self, busy: bool, message: str) -> None:
        self.is_busy = busy
        self.status_var.set(message)
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_button.configure(state=state)

    def _project(self, lat: float, lon: float, width: int, height: int) -> tuple[float, float]:
        points = self.graph.nodes
        lats = [coords[0] for coords in points.values()]
        lons = [coords[1] for coords in points.values()]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        pad_left, pad_top, pad_right, pad_bottom = self.map_padding_px
        usable_w = max(10, width - (pad_left + pad_right))
        usable_h = max(10, height - (pad_top + pad_bottom))

        lon_span = max(max_lon - min_lon, 1e-9)
        lat_span = max(max_lat - min_lat, 1e-9)

        x = pad_left + ((lon - min_lon) / lon_span) * usable_w
        # invert y so north is up
        y = pad_top + ((max_lat - lat) / lat_span) * usable_h
        return x, y

    def _node_pixels(self) -> dict[int, tuple[float, float]]:
        width, height = self._map_draw_size()

        pixels: dict[int, tuple[float, float]] = {}
        for node_id, (lat, lon) in self.graph.nodes.items():
            pixels[node_id] = self._project(lat, lon, width, height)
        return pixels

    def _map_draw_size(self) -> tuple[int, int]:
        self.canvas.update_idletasks()
        viewport_w = max(600, self.canvas.winfo_width())
        viewport_h = max(450, self.canvas.winfo_height())
        width = max(self.min_map_width, int(viewport_w * self.map_zoom))
        height = max(self.min_map_height, int(viewport_h * self.map_zoom))
        return width, height

    def _draw_base_graph(self) -> None:
        self.canvas.delete("all")
        width, height = self._map_draw_size()
        self.canvas.configure(scrollregion=(0, 0, width, height))

        px = self._node_pixels()

        for src, targets in self.graph.edges.items():
            x1, y1 = px[src]
            for dst in targets:
                if dst not in px:
                    continue
                x2, y2 = px[dst]
                self.canvas.create_line(x1, y1, x2, y2, fill="#8a8a8a", width=1)

        for node_id, (x, y) in px.items():
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ff4fc3", outline="#5c2f55", width=1)
            self.canvas.create_text(x + 8, y - 8, text=str(node_id), anchor=tk.NW, fill="#1a1a1a", font=("Segoe UI", 8, "bold"))

        try:
            origin = int(self.origin_var.get())
            if origin in px:
                x, y = px[origin]
                self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#d8f3df", outline="#0a7d34", width=2)
                self.canvas.create_text(x + 10, y + 10, text="O", fill="#0a7d34", anchor=tk.NW, font=("Segoe UI", 9, "bold"))
        except Exception:
            pass

        try:
            destination = int(self.destination_var.get())
            if destination in px:
                x, y = px[destination]
                self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#f8d7da", outline="#b30000", width=2)
                self.canvas.create_text(x + 10, y + 10, text="D", fill="#b30000", anchor=tk.NW, font=("Segoe UI", 9, "bold"))
        except Exception:
            pass

    def _clear_current_routes(self) -> None:
        self._current_routes = None
        self._hover_route_idx = None
        self._current_route_count = 0

    def _ensure_default_split(self, _event: tk.Event | None = None) -> None:
        if self._default_split_applied:
            return
        try:
            total_w = self.main_pane.winfo_width()
            if total_w < 200:
                return
            # 30% output pane, 70% map pane.
            self.main_pane.sashpos(0, int(total_w * 0.30))
            self._default_split_applied = True
        except Exception:
            return

    def on_map_wheel(self, event: tk.Event) -> str:
        if getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "num", None) == 5:
            direction = -1
        else:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1

        old_zoom = self.map_zoom
        factor = ZOOM_STEP if direction > 0 else (1.0 / ZOOM_STEP)
        new_zoom = max(MIN_MAP_ZOOM, min(MAX_MAP_ZOOM, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return "break"

        old_w, old_h = self._map_draw_size()
        anchor_x = float(self.canvas.canvasx(event.x))
        anchor_y = float(self.canvas.canvasy(event.y))
        rx = anchor_x / max(float(old_w), 1.0)
        ry = anchor_y / max(float(old_h), 1.0)

        self.map_zoom = new_zoom
        if self._current_routes:
            self._draw_routes(self._current_routes)
        else:
            self._draw_base_graph()

        new_w, new_h = self._map_draw_size()
        target_x = rx * new_w
        target_y = ry * new_h
        self.canvas.xview_moveto(max(0.0, min(1.0, (target_x - event.x) / max(float(new_w), 1.0))))
        self.canvas.yview_moveto(max(0.0, min(1.0, (target_y - event.y) / max(float(new_h), 1.0))))

        self.zoom_var.set(f"Zoom: {self.map_zoom:.2f}x")
        return "break"

    def _pick_nearest_node(self, clicked_x: float, clicked_y: float, show_miss_status: bool = True) -> int | None:
        px = self._node_pixels()
        if not px:
            return None

        nearest_node = min(
            px,
            key=lambda nid: (px[nid][0] - clicked_x) ** 2 + (px[nid][1] - clicked_y) ** 2,
        )
        dist_sq = (px[nearest_node][0] - clicked_x) ** 2 + (px[nearest_node][1] - clicked_y) ** 2
        if dist_sq > (self.pick_radius_px ** 2):
            if show_miss_status:
                self.status_var.set("Click closer to a node marker to select it.")
            return None
        return nearest_node

    def on_left_press(self, event: tk.Event) -> None:
        if self.is_busy:
            return

        clicked_x = float(self.canvas.canvasx(event.x))
        clicked_y = float(self.canvas.canvasy(event.y))
        nearest_node = self._pick_nearest_node(clicked_x, clicked_y, show_miss_status=False)
        self._pressed_origin_node = nearest_node
        self._is_dragging_map = nearest_node is None

        if self._is_dragging_map:
            self.canvas.scan_mark(event.x, event.y)
            self.status_var.set("Dragging map...")

    def on_left_drag(self, event: tk.Event) -> None:
        if self.is_busy or not self._is_dragging_map:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_left_release(self, _event: tk.Event) -> None:
        if self.is_busy:
            return

        if self._is_dragging_map:
            self._is_dragging_map = False
            self._pressed_origin_node = None
            self.status_var.set("Map moved. Left-click node for origin, right-click node for destination.")
            return

        nearest_node = self._pressed_origin_node
        self._pressed_origin_node = None
        if nearest_node is None:
            self.status_var.set("Left-click a node for origin, or drag empty space to pan.")
            return

        self.origin_var.set(str(nearest_node))
        self.output.insert(tk.END, f"Picked origin (left-click): {nearest_node}\n")
        self.status_var.set("Origin selected. Right-click a node to set destination.")

        self._clear_current_routes()
        self._draw_base_graph()

    def on_map_right_click(self, event: tk.Event) -> None:
        if self.is_busy:
            return

        clicked_x = float(self.canvas.canvasx(event.x))
        clicked_y = float(self.canvas.canvasy(event.y))
        nearest_node = self._pick_nearest_node(clicked_x, clicked_y)
        if nearest_node is None:
            return

        self.destination_var.set(str(nearest_node))
        self.output.insert(tk.END, f"Picked destination (right-click): {nearest_node}\n")
        self.status_var.set("Destination selected. Click Find Path.")
        self._clear_current_routes()
        self._draw_base_graph()

    def _draw_routes(self, routes) -> None:
        self._current_routes = routes
        self._current_route_count = len(routes)
        self._draw_base_graph()
        px = self._node_pixels()

        for idx, route in enumerate(routes):
            color = self._route_colors[idx % len(self._route_colors)]
            line_tag = f"route_line_{idx}"
            node_tag = f"route_node_{idx}"
            path = route.path
            for a, b in zip(path, path[1:]):
                if a not in px or b not in px:
                    continue
                x1, y1 = px[a]
                x2, y2 = px[b]
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=3, tags=(line_tag, "route_line"))

            for node_id in path:
                if node_id not in px:
                    continue
                x, y = px[node_id]
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="", tags=(node_tag, "route_node"))

        # simple legend
        y = 16
        for idx, route in enumerate(routes[: len(self._route_colors)]):
            color = self._route_colors[idx % len(self._route_colors)]
            mins = route.total_seconds / 60.0
            tag = f"legend_route_{idx}"
            line_tag = f"legend_line_{idx}"
            text_tag = f"legend_text_{idx}"
            self.canvas.create_line(18, y, 46, y, fill=color, width=3, tags=(tag, "legend", line_tag))
            self.canvas.create_text(
                52,
                y,
                anchor=tk.W,
                text=f"Route {idx + 1}: {mins:.2f} min",
                fill="#111",
                font=("Segoe UI", 9),
                tags=(tag, "legend", text_tag),
            )
            self.canvas.tag_bind(tag, "<Enter>", lambda _e, i=idx: self._on_legend_hover(i))
            self.canvas.tag_bind(tag, "<Leave>", lambda _e: self._on_legend_leave())
            y += 18

        self._apply_route_highlight()

    def _apply_route_highlight(self) -> None:
        for idx in range(self._current_route_count):
            color = self._route_colors[idx % len(self._route_colors)]
            is_hovered = self._hover_route_idx is not None and self._hover_route_idx == idx
            is_dimmed = self._hover_route_idx is not None and self._hover_route_idx != idx

            line_color = color if not is_dimmed else "#c8c8c8"
            node_color = color if not is_dimmed else "#c8c8c8"
            line_width = 5 if is_hovered else (2 if is_dimmed else 3)

            self.canvas.itemconfigure(f"route_line_{idx}", fill=line_color, width=line_width)
            self.canvas.itemconfigure(f"route_node_{idx}", fill=node_color)

            legend_color = color if not is_dimmed else "#b8b8b8"
            legend_width = 5 if is_hovered else 3
            legend_text_color = "#111" if not is_dimmed else "#888"
            self.canvas.itemconfigure(f"legend_line_{idx}", fill=legend_color, width=legend_width)
            self.canvas.itemconfigure(f"legend_text_{idx}", fill=legend_text_color)

    def _on_legend_hover(self, idx: int) -> None:
        if self._current_routes is None:
            return
        self._hover_route_idx = idx
        self._apply_route_highlight()

    def _on_legend_leave(self) -> None:
        if self._current_routes is None:
            return
        self._hover_route_idx = None
        self._apply_route_highlight()

    def on_run(self) -> None:
        if self.is_busy:
            return

        self.output.delete("1.0", tk.END)
        self._clear_current_routes()
        try:
            origin = int(self.origin_var.get())
            destination = int(self.destination_var.get())
            top_k = int(self.topk_var.get())
            model = self.model_var.get().strip().lower()
            algorithm = self.algorithm_var.get().strip().upper()
            hour_of_day = max(0, min(23, int(self.hour_var.get())))

            cache_key = (origin, destination, top_k, model, algorithm, hour_of_day)
            if cache_key in self._route_cache:
                self.output.insert(tk.END, "Using cached result for this request.\n")
                self._render_routes_result(
                    routes=self._route_cache[cache_key],
                    origin=origin,
                    destination=destination,
                    model=model,
                    algorithm=algorithm,
                    hour_of_day=hour_of_day,
                )
                return

            self._set_busy(True, "Preparing models and computing routes... this can take a moment.")

            threading.Thread(
                target=self._compute_routes_worker,
                args=(origin, destination, top_k, model, algorithm, hour_of_day, cache_key),
                daemon=True,
            ).start()
        except Exception as exc:  # pragma: no cover
            self.output.insert(tk.END, f"Error: {exc}\n")
            self._set_busy(False, "Input error. Please check values and try again.")

    def _compute_routes_worker(
        self,
        origin: int,
        destination: int,
        top_k: int,
        model: str,
        algorithm: str,
        hour_of_day: int,
        cache_key: tuple[int, int, int, str, str, int],
    ) -> None:
        try:
            with self._ctx_lock:
                if self.ctx is None:
                    self.ctx = build_context(DEFAULT_CONFIG_PATH)

            routes = recommend_routes(
                ctx=self.ctx,
                origin=origin,
                destination=destination,
                top_k=top_k,
                model_name=model,
                path_method=algorithm,
                hour_of_day=hour_of_day,
            )
            self._route_cache[cache_key] = routes
            self.root.after(
                0,
                lambda: self._render_routes_result(
                    routes=routes,
                    origin=origin,
                    destination=destination,
                    model=model,
                    algorithm=algorithm,
                    hour_of_day=hour_of_day,
                ),
            )
        except Exception as exc:  # pragma: no cover
            msg = str(exc)
            self.root.after(0, lambda m=msg: self._render_error(m))

    def _render_routes_result(
        self,
        routes,
        origin: int,
        destination: int,
        model: str,
        algorithm: str,
        hour_of_day: int,
    ) -> None:
        self.output.delete("1.0", tk.END)
        if not routes:
            self.output.insert(tk.END, "No feasible route found.\n")
            self._clear_current_routes()
            self._draw_base_graph()
            self._set_busy(False, "No route found. Try different nodes/algorithm/hour.")
            return

        self.output.insert(
            tk.END,
            f"Routes from {origin} to {destination} using model={model}, algorithm={algorithm}, hour={hour_of_day}:\n\n",
        )
        for idx, route in enumerate(routes, start=1):
            nodes = " -> ".join(str(n) for n in route.path)
            self.output.insert(
                tk.END,
                f"{idx}. Time: {route.total_seconds / 60.0:.2f} min\n"
                f"   Path: {nodes}\n\n",
            )
        self._draw_routes(routes)
        self._set_busy(False, "Routes updated.")

    def _render_error(self, message: str) -> None:
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"Error: {message}\n")
        self._set_busy(False, "Failed to compute routes. Please retry.")


def main() -> None:
    root = tk.Tk()
    app = TBRGSApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
