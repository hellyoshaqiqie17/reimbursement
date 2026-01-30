# Receipt OCR API Documentation

Base URL: `https://reimbursement-544676101248.asia-southeast2.run.app`

## 1. Health Check
Check if the API service is running.

- **Endpoint**: `GET /api/v1/health`
- **Response**:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "ocr_engine": "paddleocr"
}
```

## 2. Extract Receipt Data
Upload a receipt image to extract structured data.

- **Endpoint**: `POST /api/v1/extract`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
    - `file` (Required, File): The receipt image file (JPG, PNG, WEBP, etc).
    - `debug` (Optional, Query Param): Set to `true` to receive raw OCR text and processing time.

### Success Response (200 OK)
```json
{
    "merchant_name": "PERTAMINA",
    "transaction_date": "2023-06-24",
    "total_amount_raw": "30,000",
    "total_amount_value": 30000.0,
    "confidence_score": 0.893
}
```

### Response Fields:
- `merchant_name`: Detected name of the vendor/store (e.g., "Starbucks", "Pertamina"). Returns `null` if not found.
- `transaction_date`: Date in `YYYY-MM-DD` format. Returns `null` if not found.
- `total_amount_raw`: The raw string of the total amount found on receipt (e.g., "Rp 50.000").
- `total_amount_value`: The parsed numeric value (float) of the total amount (e.g., `50000.0`). Ready for calculation.
- `confidence_score`: Average confidence of the OCR detection (0.0 - 1.0).

### Error Response (400 Bad Request)
```json
{
    "detail": {
        "error": "InvalidImageError",
        "message": "Failed to load image..."
    }
}
```

## Integration Example (JavaScript/Axios)

```javascript
const formData = new FormData();
formData.append('file', imageFile); // imageFile from <input type="file">

try {
  const response = await axios.post('https://reimbursement-544676101248.asia-southeast2.run.app/api/v1/extract', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  
  const { merchant_name, total_amount_value, transaction_date } = response.data;
  console.log(`Total: ${total_amount_value}`);
  
} catch (error) {
  console.error('OCR Failed:', error);
}
```
