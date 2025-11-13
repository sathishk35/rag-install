#!/bin/bash
# RAG System Standalone Verification Script
# Quick verification for standalone deployment

set -e

# Configuration
API_BASE="http://localhost:8000"
OLLAMA_BASE="http://localhost:11434"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

header() {
    echo -e "${PURPLE}"
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo -e "${NC}"
}

test_start() {
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -n "Testing $1... "
}

test_pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo -e "${GREEN}PASS${NC}"
    if [[ -n "$1" ]]; then
        echo "  ✓ $1"
    fi
}

test_fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}FAIL${NC}"
    if [[ -n "$1" ]]; then
        echo "  ✗ $1"
    fi
}

# Test standalone services
test_standalone_services() {
    header "TESTING STANDALONE SERVICES"
    
    services=("postgresql" "redis-rag" "qdrant" "ollama" "rag-api-standalone")
    
    for service in "${services[@]}"; do
        test_start "service $service"
        if systemctl is-active --quiet "$service"; then
            test_pass "Service is running"
        else
            test_fail "Service is not running"
        fi
    done
}

# Test basic connectivity
test_connectivity() {
    header "TESTING BASIC CONNECTIVITY"
    
    # Test PostgreSQL
    test_start "PostgreSQL connection"
    if sudo -u rag-system psql -h localhost rag_metadata -c "SELECT 1;" >/dev/null 2>&1; then
        test_pass "Database connection successful"
    else
        test_fail "Cannot connect to database"
    fi
    
    # Test Redis
    test_start "Redis connection"
    if redis-cli -p 6380 ping 2>/dev/null | grep -q PONG; then
        test_pass "Redis connection successful"
    else
        test_fail "Cannot connect to Redis"
    fi
    
    # Test Qdrant
    test_start "Qdrant connection"
    if curl -s --max-time 5 http://localhost:6333/collections >/dev/null; then
        collections=$(curl -s http://localhost:6333/collections | jq '.result.collections | length' 2>/dev/null || echo "0")
        test_pass "Qdrant responding with $collections collections"
    else
        test_fail "Cannot connect to Qdrant"
    fi
    
    # Test Ollama
    test_start "Ollama connection"
    if curl -s --max-time 5 "$OLLAMA_BASE/api/tags" >/dev/null; then
        models=$(curl -s "$OLLAMA_BASE/api/tags" | jq '.models | length' 2>/dev/null || echo "0")
        test_pass "Ollama responding with $models models"
    else
        test_fail "Cannot connect to Ollama"
    fi
    
    # Test RAG API
    test_start "RAG API health"
    health_response=$(curl -s --max-time 10 "$API_BASE/api/health" 2>/dev/null)
    if echo "$health_response" | grep -q '"status"'; then
        status=$(echo "$health_response" | jq -r '.status' 2>/dev/null || echo "unknown")
        test_pass "API responding with status: $status"
    else
        test_fail "API not responding"
    fi
}

# Test RAG functionality
test_rag_functionality() {
    header "TESTING RAG FUNCTIONALITY"
    
    # Test simple query
    test_start "simple query processing"
    simple_response=$(curl -s --max-time 15 -X POST "$API_BASE/api/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "Hello, please respond", "user_id": "test_user"}' 2>/dev/null)
    
    if echo "$simple_response" | grep -q '"response"'; then
        response_length=$(echo "$simple_response" | jq -r '.response' | wc -c)
        test_pass "Query processed successfully (${response_length} chars)"
    else
        test_fail "Query processing failed"
    fi
    
    # Test technical query
    test_start "technical query processing"
    tech_response=$(curl -s --max-time 20 -X POST "$API_BASE/api/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "What is GPIO in embedded systems?", "user_id": "test_user"}' 2>/dev/null)
    
    if echo "$tech_response" | grep -q '"response"'; then
        confidence=$(echo "$tech_response" | jq -r '.confidence_score // 0' 2>/dev/null)
        test_pass "Technical query processed (confidence: $confidence)"
    else
        test_fail "Technical query processing failed"
    fi
    
    # Test document stats
    test_start "document statistics"
    stats_response=$(curl -s --max-time 10 "$API_BASE/api/documents/stats" 2>/dev/null)
    if echo "$stats_response" | grep -q '"documents"'; then
        test_pass "Document statistics accessible"
    else
        test_fail "Document statistics not accessible"
    fi
    
    # Test models status
    test_start "models status"
    models_response=$(curl -s --max-time 10 "$API_BASE/api/models/status" 2>/dev/null)
    if echo "$models_response" | grep -q '"models"'; then
        embedding_models=$(echo "$models_response" | jq -r '.embedding_models | length' 2>/dev/null || echo "0")
        test_pass "Models status available ($embedding_models embedding models)"
    else
        test_fail "Models status not available"
    fi
}

# Test document processing
test_document_processing() {
    header "TESTING DOCUMENT PROCESSING"
    
    # Create test document
    test_doc="/tmp/rag_standalone_test.txt"
    cat > "$test_doc" << 'EOF'
Test Document for Standalone RAG System

This is a simple test document to verify document processing in standalone mode.

Key Topics:
1. GPIO Programming - Digital input/output control
2. UART Communication - Serial data transmission
3. Embedded Systems - Microcontroller programming

Example Code:
```c
void gpio_init() {
    // Initialize GPIO pin
    GPIO_Init(GPIOA, GPIO_PIN_5, GPIO_MODE_OUTPUT);
}
```
EOF
    
    # Test document upload
    test_start "document upload"
    upload_response=$(curl -s --max-time 30 -X POST "$API_BASE/api/documents/upload" \
        -F "file=@$test_doc" \
        -F "security_classification=internal" \
        -F "domain=general" 2>/dev/null)
    
    if echo "$upload_response" | grep -q '"success": true'; then
        test_pass "Document upload successful"
    else
        test_fail "Document upload failed"
    fi
    
    # Clean up test document
    rm -f "$test_doc"
    
    # Test document processor service
    test_start "document processor service"
    if systemctl is-active --quiet rag-processor-standalone; then
        test_pass "Document processor is running"
    else
        test_fail "Document processor is not running"
    fi
}

# Test system performance
test_performance() {
    header "TESTING SYSTEM PERFORMANCE"
    
    # Test API response time
    test_start "API response time"
    start_time=$(date +%s.%N)
    curl -s --max-time 10 "$API_BASE/api/health" >/dev/null 2>&1
    end_time=$(date +%s.%N)
    response_time=$(echo "$end_time - $start_time" | bc)
    
    if (( $(echo "$response_time < 2.0" | bc -l) )); then
        test_pass "Response time: ${response_time}s"
    else
        test_fail "Slow response time: ${response_time}s"
    fi
    
    # Test query processing time
    test_start "query processing time"
    start_time=$(date +%s.%N)
    curl -s --max-time 30 -X POST "$API_BASE/api/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "test performance", "user_id": "test_user"}' >/dev/null 2>&1
    end_time=$(date +%s.%N)
    query_time=$(echo "$end_time - $start_time" | bc)
    
    if (( $(echo "$query_time < 15.0" | bc -l) )); then
        test_pass "Query time: ${query_time}s"
    else
        test_fail "Slow query time: ${query_time}s"
    fi
    
    # Test system resources
    test_start "system resources"
    memory_percent=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
    disk_percent=$(df / | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{print $5}' | sed 's/%//')
    
    if [[ $memory_percent -lt 85 ]] && [[ $disk_percent -lt 90 ]]; then
        test_pass "Memory: ${memory_percent}%, Disk: ${disk_percent}%"
    else
        test_fail "High resource usage - Memory: ${memory_percent}%, Disk: ${disk_percent}%"
    fi
}

# Test Ollama integration
test_ollama_integration() {
    header "TESTING OLLAMA INTEGRATION"
    
    # Test Ollama models
    test_start "Ollama models list"
    models_response=$(curl -s --max-time 10 "$OLLAMA_BASE/api/tags" 2>/dev/null)
    if echo "$models_response" | jq -e '.models' >/dev/null 2>&1; then
        model_count=$(echo "$models_response" | jq '.models | length')
        model_names=$(echo "$models_response" | jq -r '.models[].name' | head -3 | tr '\n' ', ' | sed 's/,$//')
        test_pass "Found $model_count models: $model_names"
    else
        test_fail "Cannot retrieve Ollama models"
    fi
    
    # Check for DeepSeek model
    test_start "DeepSeek model availability"
    if echo "$models_response" | grep -q "deepseek"; then
        deepseek_model=$(echo "$models_response" | jq -r '.models[] | select(.name | contains("deepseek")) | .name' | head -1)
        test_pass "DeepSeek model available: $deepseek_model"
    else
        test_fail "DeepSeek model not found (may need: ollama pull deepseek-r1:70b)"
    fi
    
    # Test direct Ollama generation
    test_start "direct Ollama generation"
    ollama_response=$(curl -s --max-time 20 -X POST "$OLLAMA_BASE/api/generate" \
        -H "Content-Type: application/json" \
        -d '{"model": "deepseek-r1:70b", "prompt": "Hello", "stream": false}' 2>/dev/null)
    
    if echo "$ollama_response" | grep -q '"response"'; then
        test_pass "Direct Ollama generation working"
    else
        test_fail "Direct Ollama generation failed"
    fi
}

# Test data integrity
test_data_integrity() {
    header "TESTING DATA INTEGRITY"
    
    # Test database tables
    test_start "database tables"
    table_count=$(sudo -u rag-system psql -h localhost rag_metadata -t -c "
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name IN ('documents', 'chunks', 'user_access');
    " 2>/dev/null | tr -d ' ')
    
    if [[ "$table_count" == "3" ]]; then
        test_pass "All required tables exist"
    else
        test_fail "Missing database tables (found: $table_count/3)"
    fi
    
    # Test vector collections
    test_start "vector collections"
    collections_response=$(curl -s --max-time 5 http://localhost:6333/collections 2>/dev/null)
    if echo "$collections_response" | jq -e '.result.collections' >/dev/null 2>&1; then
        collection_count=$(echo "$collections_response" | jq '.result.collections | length')
        test_pass "Vector collections available: $collection_count"
    else
        test_fail "Vector collections not accessible"
    fi
    
    # Test user access
    test_start "user access data"
    user_count=$(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM user_access;" 2>/dev/null | tr -d ' ')
    if [[ "$user_count" =~ ^[0-9]+$ ]] && [[ $user_count -gt 0 ]]; then
        test_pass "User access data available: $user_count users"
    else
        test_fail "No user access data found"
    fi
}

# Generate test report
generate_test_report() {
    header "GENERATING TEST REPORT"
    
    report_file="/opt/rag-system/STANDALONE_VERIFICATION_REPORT.md"
    
    cat > "$report_file" << EOF
# RAG System Standalone Verification Report

**Generated:** $(date)
**Mode:** Standalone Testing
**Test Results:** $TESTS_PASSED/$TESTS_TOTAL tests passed

## Test Summary

$(if [[ $TESTS_FAILED -eq 0 ]]; then
    echo "✅ **ALL TESTS PASSED** - System is fully functional"
else
    echo "⚠️ **$TESTS_FAILED TESTS FAILED** - Review failed tests below"
fi)

- **Success Rate:** $(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)%
- **Total Tests:** $TESTS_TOTAL
- **Passed:** $TESTS_PASSED
- **Failed:** $TESTS_FAILED

## System Status

### Services
$(for service in postgresql redis-rag qdrant ollama rag-api-standalone rag-processor-standalone rag-monitor-standalone; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        echo "- ✅ $service: Running"
    else
        echo "- ❌ $service: Not Running"
    fi
done)

### API Endpoints
- **Health Check:** $(curl -s --max-time 5 "$API_BASE/api/health" >/dev/null && echo "✅ Working" || echo "❌ Failed")
- **Query Processing:** $(curl -s --max-time 10 -X POST "$API_BASE/api/query" -H "Content-Type: application/json" -d '{"query": "test", "user_id": "test"}' >/dev/null && echo "✅ Working" || echo "❌ Failed")
- **Document Upload:** $(curl -s --max-time 10 "$API_BASE/api/documents/stats" >/dev/null && echo "✅ Working" || echo "❌ Failed")
- **Models Status:** $(curl -s --max-time 5 "$API_BASE/api/models/status" >/dev/null && echo "✅ Working" || echo "❌ Failed")

### Database Status
- **Documents:** $(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM documents;" 2>/dev/null | tr -d ' ' || echo "N/A")
- **Chunks:** $(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM chunks;" 2>/dev/null | tr -d ' ' || echo "N/A")
- **Users:** $(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM user_access;" 2>/dev/null | tr -d ' ' || echo "N/A")

### Ollama Integration
- **Service Status:** $(systemctl is-active --quiet ollama && echo "✅ Running" || echo "❌ Not Running")
- **API Access:** $(curl -s --max-time 5 "$OLLAMA_BASE/api/tags" >/dev/null && echo "✅ Working" || echo "❌ Failed")
- **Models Available:** $(curl -s "$OLLAMA_BASE/api/tags" 2>/dev/null | jq '.models | length' 2>/dev/null || echo "0")
- **DeepSeek Model:** $(curl -s "$OLLAMA_BASE/api/tags" 2>/dev/null | grep -q "deepseek" && echo "✅ Available" || echo "❌ Missing")

## Performance Metrics

### Response Times
- **API Health:** $(curl -s -w "%{time_total}" --max-time 5 "$API_BASE/api/health" -o /dev/null 2>/dev/null || echo "N/A")s
- **Simple Query:** $(curl -s -w "%{time_total}" --max-time 15 -X POST "$API_BASE/api/query" -H "Content-Type: application/json" -d '{"query": "test", "user_id": "test"}' -o /dev/null 2>/dev/null || echo "N/A")s

### System Resources
- **Memory Usage:** $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')
- **Disk Usage:** $(df / | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{print $5}')
- **CPU Load:** $(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | sed 's/,//')

$(if command -v nvidia-smi >/dev/null 2>&1; then
    echo "### GPU Status"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "- GPU: $line"
    done
fi)

## Quick Test Commands

\`\`\`bash
# Run quick verification
sudo /opt/rag-system/scripts/verify_standalone.sh

# Test API directly
curl -X POST http://localhost:8000/api/query \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is GPIO?", "user_id": "test_user"}'

# Check system status
sudo /opt/rag-system/scripts/rag-standalone.sh status

# View API documentation
open http://localhost:8000/api/docs

# Run demonstration queries
sudo /opt/rag-system/scripts/rag-standalone.sh demo
\`\`\`

## Troubleshooting

$(if [[ $TESTS_FAILED -gt 0 ]]; then
    echo "### Issues Found"
    echo "Review the test output above for specific failures."
    echo ""
    echo "Common solutions:"
    echo "- **Service not running:** \`sudo systemctl start SERVICE_NAME\`"
    echo "- **API not responding:** Check logs with \`sudo journalctl -u rag-api-standalone -f\`"
    echo "- **Ollama issues:** Ensure DeepSeek model is installed: \`ollama pull deepseek-r1:70b\`"
    echo "- **Database issues:** Check PostgreSQL status: \`sudo systemctl status postgresql\`"
    echo ""
fi)

### Log Files
- **API Logs:** \`sudo journalctl -u rag-api-standalone -f\`
- **Processor Logs:** \`sudo journalctl -u rag-processor-standalone -f\`
- **Monitor Logs:** \`sudo journalctl -u rag-monitor-standalone -f\`
- **System Logs:** \`/var/log/rag/\`

### Service Control
- **Start All:** \`sudo /opt/rag-system/scripts/rag-standalone.sh start\`
- **Stop All:** \`sudo /opt/rag-system/scripts/rag-standalone.sh stop\`
- **Restart:** \`sudo /opt/rag-system/scripts/rag-standalone.sh restart\`

## Next Steps

$(if [[ $TESTS_FAILED -eq 0 ]]; then
    echo "✅ **System Ready!** The standalone RAG system is fully functional."
    echo ""
    echo "Recommended next steps:"
    echo "1. Upload your company documents for indexing"
    echo "2. Create additional users with appropriate security levels"
    echo "3. Test with domain-specific queries"
    echo "4. Monitor system performance under load"
    echo "5. When ready, proceed with OpenWebUI integration"
else
    echo "⚠️ **Issues Found** - Address the failed tests before proceeding:"
    echo ""
    echo "1. Review and fix failed services"
    echo "2. Check system logs for errors"
    echo "3. Verify all dependencies are installed"
    echo "4. Re-run verification after fixes"
fi)

---
**Report Generated:** $(date)
**System:** $(hostname) - $(uname -a)
EOF

    chown rag-system:rag-system "$report_file"
    log "Verification report generated: $report_file"
}

# Main execution
main() {
    header "RAG SYSTEM STANDALONE VERIFICATION"
    info "Running comprehensive verification of standalone deployment"
    info "This will test all components and generate a detailed report"
    echo
    
    # Run all tests
    test_standalone_services
    test_connectivity
    test_rag_functionality
    test_document_processing
    test_performance
    test_ollama_integration
    test_data_integrity
    
    # Generate report
    generate_test_report
    
    # Final summary
    header "VERIFICATION COMPLETE"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log "🎉 All $TESTS_TOTAL tests passed! Standalone RAG system is fully functional."
        log ""
        log "✅ System Status: HEALTHY"
        log "✅ API Status: OPERATIONAL" 
        log "✅ Ollama Integration: WORKING"
        log "✅ Database Status: CONNECTED"
        log "✅ Performance: GOOD"
        log ""
        log "🚀 The standalone RAG system is ready for use!"
        echo
        log "🧪 Try these test commands:"
        info "  # Quick test"
        info "  curl -X POST http://localhost:8000/api/query \\"
        info "    -H 'Content-Type: application/json' \\"
        info "    -d '{\"query\": \"What is GPIO?\", \"user_id\": \"test_user\"}'"
        echo
        info "  # Run demonstrations"
        info "  sudo /opt/rag-system/scripts/rag-standalone.sh demo"
        echo
        info "  # View API documentation"
        info "  http://localhost:8000/api/docs"
    else
        warn "⚠️  $TESTS_FAILED out of $TESTS_TOTAL tests failed."
        warn "Success Rate: $(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)%"
        warn ""
        warn "Please review the verification report and address any issues:"
        warn "  /opt/rag-system/STANDALONE_VERIFICATION_REPORT.md"
        warn ""
        warn "Common fixes:"
        warn "  • Check service status: sudo /opt/rag-system/scripts/rag-standalone.sh status"
        warn "  • View logs: sudo journalctl -u rag-api-standalone -f"
        warn "  • Install DeepSeek model: ollama pull deepseek-r1:70b"
    fi
    
    echo
    log "📋 Access Points:"
    log "   • RAG API: http://localhost:8000"
    log "   • API Documentation: http://localhost:8000/api/docs"
    log "   • Health Check: http://localhost:8000/api/health"
    log ""
    log "📄 Reports:"
    log "   • Verification Report: /opt/rag-system/STANDALONE_VERIFICATION_REPORT.md"
    log "   • Deployment Report: /opt/rag-system/STANDALONE_DEPLOYMENT_REPORT.md"
    log ""
    log "🔧 Management:"
    log "   • Control Script: /opt/rag-system/scripts/rag-standalone.sh"
    log "   • Quick Test: /opt/rag-system/scripts/quick_test.sh"
    
    # Exit with appropriate code
    if [[ $TESTS_FAILED -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"