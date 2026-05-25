from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from tbrgs.pipeline import build_context, recommend_routes


class TBRGSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TBRGS - Assignment 2B")
        self.root.geometry("900x600")

        self.ctx = build_context("config/tbrgs_defaults.json")

        runtime = self.ctx.config["runtime"]

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

        ttk.Button(form, text="Recommend Routes", command=self.on_run).grid(row=0, column=8, sticky=tk.W, padx=8, pady=4)

        self.output = tk.Text(root, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.output.insert(tk.END, "Ready. Configure inputs and click Recommend Routes.\n")

    def on_run(self) -> None:
        self.output.delete("1.0", tk.END)
        try:
            origin = int(self.origin_var.get())
            destination = int(self.destination_var.get())
            top_k = int(self.topk_var.get())
            model = self.model_var.get().strip().lower()

            routes = recommend_routes(
                ctx=self.ctx,
                origin=origin,
                destination=destination,
                top_k=top_k,
                model_name=model,
            )
            if not routes:
                self.output.insert(tk.END, "No feasible route found.\n")
                return

            self.output.insert(tk.END, f"Routes from {origin} to {destination} using {model}:\n\n")
            for idx, route in enumerate(routes, start=1):
                nodes = " -> ".join(str(n) for n in route.path)
                self.output.insert(
                    tk.END,
                    f"{idx}. Time: {route.total_seconds / 60.0:.2f} min\n"
                    f"   Path: {nodes}\n\n",
                )
        except Exception as exc:  # pragma: no cover
            self.output.insert(tk.END, f"Error: {exc}\n")


def main() -> None:
    root = tk.Tk()
    app = TBRGSApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
