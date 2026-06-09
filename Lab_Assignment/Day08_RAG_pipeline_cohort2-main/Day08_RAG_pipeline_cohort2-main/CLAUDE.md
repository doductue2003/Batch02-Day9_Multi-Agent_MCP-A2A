# Cấu hình dự án (CLAUDE.md)

## 1. Quy tắc hoạt động của AI Agent (Bắt buộc)
- Với mỗi task được giao, trước khi viết code, BẮT BUỘC phải kiểm tra danh sách các kỹ năng tại thư mục global `C:\Users\V3400\.gemini\antigravity-ide\skills\`.
- Hãy tự động gọi công cụ `view_file` để đọc file `SKILL.md` tương ứng nhằm áp dụng chính xác các mẫu thiết kế (design patterns) và quy trình kiểm thử (TDD).

## 2. Các lệnh chạy dự án (Commands)
*Hãy sử dụng các lệnh dưới đây khi kiểm tra chất lượng hoặc chạy code:*
- **Cài đặt thư viện:** `npm install` hoặc `pip install -r requirements.txt` (tùy ngôn ngữ)
- **Chạy Unit Test:** `npm run test` hoặc `pytest`
- **Chạy E2E Test:** `npx playwright test`
- **Chạy Linter/Formatter:** `npm run lint` hoặc `flake8`
- **Khởi động Local Server:** `npm run dev` hoặc `uvicorn main:app --reload`

## 3. Kiến trúc dự án & Coding Style (Preferred Tech Stack)
*Dự án này tuân thủ các chuẩn công nghệ và phong cách sau:*
- **Backend:** FastAPI (Python) hoặc Next.js API Routes (TypeScript).
- **Frontend:** React / Next.js (App Router), CSS Vanilla / Tailwind CSS.
- **Database:** PostgreSQL (sử dụng Prisma/Drizzle ORM để viết migrations) + Redis.
- **Quy tắc viết code:** Luôn sử dụng cơ chế xử lý bất đồng bộ (async/await), luôn kiểm tra và validate đầu vào của người dùng bằng schemas (như Zod hoặc Pydantic) để tránh Prompt Injection / SQL Injection.
