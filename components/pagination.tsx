import { ChevronLeft, ChevronRight } from "lucide-react";

type PageItem = number | "ellipsis";

function buildPageItems(page: number, totalPages: number): PageItem[] {
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, index) => index + 1);
  }
  const items: PageItem[] = [1];
  if (page > 4) items.push("ellipsis");
  const start = Math.max(2, page - 1);
  const end = Math.min(totalPages - 1, page + 1);
  for (let value = start; value <= end; value += 1) items.push(value);
  if (page < totalPages - 3) items.push("ellipsis");
  items.push(totalPages);
  return items;
}

export function Pagination({
  page,
  totalPages,
  total,
  unit,
  onPageChange,
}: {
  page: number;
  totalPages: number;
  total: number;
  unit: string;
  onPageChange: (page: number) => void;
}) {
  const pages = Math.max(1, totalPages);
  return (
    <div className="pagination">
      <span>共 {total} {unit}</span>
      <div className="page-list">
        <button type="button" className="page-btn" disabled={page <= 1} aria-label="上一页" title="上一页" onClick={() => onPageChange(Math.max(1, page - 1))}><ChevronLeft size={15} aria-hidden="true" /></button>
        {buildPageItems(page, pages).map((item, index) => item === "ellipsis"
          ? <span className="page-ellipsis" key={`ellipsis-${index}`}>…</span>
          : <button type="button" key={item} className={`page-btn ${item === page ? "active" : ""}`} onClick={() => onPageChange(item)}>{item}</button>)}
        <button type="button" className="page-btn" disabled={page >= pages} aria-label="下一页" title="下一页" onClick={() => onPageChange(Math.min(pages, page + 1))}><ChevronRight size={15} aria-hidden="true" /></button>
      </div>
    </div>
  );
}
